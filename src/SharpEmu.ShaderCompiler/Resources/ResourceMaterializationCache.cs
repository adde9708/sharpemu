// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;

namespace SharpEmu.ShaderCompiler.Resources;

// Copies guest bytes only when they are already resident on the CPU side. It never
// synchronizes: a range the GPU may still own, or that a clean read would refuse,
// returns false so the caller falls back to a full materialization.
public delegate bool ResidentGuestBytesReader(ulong address, Span<byte> destination, bool clean);

// Materialization is a pure function of the plan, the draw's user data, the shader
// base, the compute state and the guest words it reads. This cache records exactly
// those words and reuses the result while every one of them is unchanged, so a draw
// that re-binds the same resources skips the descriptor walk. Any failed read, any
// GPU-owned range and any byte difference make the entry miss; nothing is guessed.
public sealed class ResourceMaterializationCache
{
    // Two generations approximate LRU: a full young generation becomes the old one,
    // and an old entry that is used again is promoted back.
    private readonly int _generationCapacity;
    private Dictionary<ulong, Entry> _young = new();
    private Dictionary<ulong, Entry> _old = new();
    private byte[] _scratch = new byte[256];
    private ReadRecorder? _spareRecorder;

    public ResourceMaterializationCache(int generationCapacity = 16384)
    {
        _generationCapacity = Math.Max(1, generationCapacity);
    }

    private static long _totalHits;
    private static long _totalMisses;
    private static long _totalUncacheable;
    // The chain absorbs user-data variation as well as guest-word variation, so it carries
    // the capacity the per-key signature list used to provide.
    private const int MaxVariants = 8;
    private static long _totalStale;
    private static long _totalStaleUnreadable;
    private static long _totalRefreshes;

    [ThreadStatic]
    private static bool _readingTable;

    public static bool ReadingTable
    {
        get => _readingTable;
        private set => _readingTable = value;
    }

    public long Hits { get; private set; }
    public long Misses { get; private set; }
    public long Uncacheable { get; private set; }
    public long TableRefreshes { get; private set; }

    // Process-wide counters since the previous call, for the periodic render report.
    public static string TakeReport()
    {
        var hits = Interlocked.Exchange(ref _totalHits, 0);
        var misses = Interlocked.Exchange(ref _totalMisses, 0);
        var uncacheable = Interlocked.Exchange(ref _totalUncacheable, 0);
        var stale = Interlocked.Exchange(ref _totalStale, 0);
        var staleUnreadable = Interlocked.Exchange(ref _totalStaleUnreadable, 0);
        var refreshes = Interlocked.Exchange(ref _totalRefreshes, 0);
        var total = hits + misses;
        return FormattableString.Invariant(
            $"[PERF][RESOURCE_CACHE] hits={hits} misses={misses} stale={stale} stale_unreadable={staleUnreadable} refreshes={refreshes} uncacheable={uncacheable} hit_rate={(total == 0 ? 0 : hits * 100.0 / total):F1}% {RawReadPrefetch.TakeReport()}");
    }

    public bool Materialize(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader,
        ref ResourceSnapshot snapshot,
        ref ResourceSpecialization specialization,
        out ResourceMaterializationFailure failure)
    {
        var key = KeyOf(plan, inputs);
        var resident = TryFind(key, out var cached);
        if (resident)
        {
            var unreadable = false;
            Entry? previous = null;
            for (var variant = cached; variant is not null; previous = variant, variant = variant.Next)
            {
                if (!variant.Matches(plan, inputs))
                    continue;
                if (Validate(variant, residentReader, out var variantUnreadable))
                {
                    if (previous is not null)
                    {
                        previous.Next = variant.Next;
                        variant.Next = cached;
                        Store(key, variant);
                    }

                    Hits++;
                    Interlocked.Increment(ref _totalHits);
                    snapshot = Rebind(variant.Snapshot, inputs);
                    specialization = variant.Specialization;
                    failure = default;
                    return true;
                }

                unreadable |= variantUnreadable;
            }

            if (TryRefreshTable(key, cached, plan, inputs, residentReader, out var refreshed))
            {
                TableRefreshes++;
                Interlocked.Increment(ref _totalRefreshes);
                snapshot = Rebind(refreshed.Snapshot, inputs);
                specialization = refreshed.Specialization;
                failure = default;
                return true;
            }

            Interlocked.Increment(ref _totalStale);
            if (unreadable)
                Interlocked.Increment(ref _totalStaleUnreadable);
        }

        Misses++;
        Interlocked.Increment(ref _totalMisses);
        var recorder = TakeRecorder();
        try
        {
            var recording = new ResourceRuntimeInputs
            {
                UserData = inputs.UserData,
                ShaderBase = inputs.ShaderBase,
                UserDataRead = recorder.RecordUserData,
                ReadMemory = recorder.Wrap(inputs.ReadMemory, clean: false),
                ReadCleanMemory = recorder.Wrap(inputs.ReadCleanMemory, clean: true),
                ReadResidentMemory = recorder.WrapResident(inputs.ReadResidentMemory),
                ReadsClean = inputs.ReadsClean,
                ComputeState = inputs.ComputeState,
                TablePhase = recorder.TablePhase,
            };
            if (!ResourceMaterializer.Materialize(plan, recording, ref snapshot, ref specialization, out failure))
                return false;

            if (recorder.Failed)
            {
                Uncacheable++;
                Interlocked.Increment(ref _totalUncacheable);
                return true;
            }

            var built = recorder.Build(plan, inputs, snapshot, specialization);
            if (resident)
            {
                // A signature that is already cached takes that entry's slot, so a recurring set
                // of push constants keeps its entry instead of evicting a different one.
                Entry? previous = null;
                for (var variant = cached; variant is not null; previous = variant, variant = variant.Next)
                {
                    if (!variant.SameSignature(built))
                        continue;
                    built.Next = variant.Next;
                    if (previous is null)
                        Store(key, built);
                    else
                    {
                        previous.Next = built;
                        Store(key, cached);
                    }
                    return true;
                }

                // A new signature goes to the front. The chain is bounded so a cycling set of
                // push constants cannot evict every other entry under the same key.
                built.Next = cached;
                var depth = 1;
                for (var variant = built; variant.Next is not null; variant = variant.Next)
                {
                    if (++depth >= MaxVariants)
                    {
                        variant.Next = null;
                        break;
                    }
                }
            }

            Store(key, built);
            return true;
        }
        finally
        {
            ReturnRecorder(recorder);
        }
    }

    // Most stale entries in Demon's Souls differ only in words the shader reads through scalar
    // loads (inline constants rewritten every frame); the descriptors, device ranges and
    // specialization are unchanged. When every changed word was read only while evaluating the
    // flattened table, only the table is evaluated again. It is refused when the plan's
    // specialization reads or extends the table (indirect or candidate tables), when the table
    // now reads a word the entry did not validate, or when a word moved between the two reads.
    private bool TryRefreshTable(ulong key, Entry cached, ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader, out Entry refreshed)
    {
        refreshed = null!;
        // The refresh re-evaluates the table from the current draw's user data but keeps the
        // recorded descriptors, so any word those descriptors depended on must still hold.
        if (!cached.TableRefreshable || !cached.UserDataHolds(inputs))
            return false;

        var current = new byte[cached.Bytes.Length];
        for (var index = 0; index < cached.RangeAddresses.Length; index++)
        {
            if (!residentReader(cached.RangeAddresses[index], current.AsSpan(cached.RangeOffsets[index], cached.RangeLengths[index]), cached.RangeClean[index]))
                return false;
        }

        var changed = false;
        for (var offset = 0; offset < current.Length; offset += sizeof(uint))
        {
            if (current.AsSpan(offset, sizeof(uint)).SequenceEqual(cached.Bytes.AsSpan(offset, sizeof(uint))))
                continue;
            if (!cached.WordTableOnly[offset / sizeof(uint)])
                return false;
            changed = true;
        }

        if (!changed)
            return false;

        var recorder = TakeRecorder();
        try
        {
            var recording = new ResourceRuntimeInputs
            {
                UserData = inputs.UserData,
                ShaderBase = inputs.ShaderBase,
                ReadMemory = recorder.Wrap(inputs.ReadMemory, clean: false),
                ReadCleanMemory = recorder.Wrap(inputs.ReadCleanMemory, clean: true),
                ReadResidentMemory = recorder.WrapResident(inputs.ReadResidentMemory),
                ReadsClean = inputs.ReadsClean,
                ComputeState = inputs.ComputeState,
                UserDataRead = recorder.RecordUserData,
            };
            var cachedTable = cached.Snapshot.FlattenedResourceTable;
            ReadingTable = true;
            bool evaluated;
            uint[] table;
            try
            {
                evaluated = ResourceMaterializer.TryEvaluateTable(plan, recording, out table);
            }
            finally
            {
                ReadingTable = false;
            }

            if (!evaluated || recorder.Failed || table.Length != cachedTable.Length)
                return false;

            foreach (var (address, word, _, _) in recorder.Reads)
            {
                if (!TryFindWord(cached, address, out var offset) ||
                    System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current.AsSpan(offset, sizeof(uint))) != word)
                    return false;
            }

            // The written device-address slots come from reads validated unchanged above.
            foreach (var slot in plan.WrittenRangeSlotByHandle.Values)
                cachedTable.AsSpan((int)slot, ShaderResourcePlan.WrittenRangeDwordCount).CopyTo(table.AsSpan((int)slot));

            var previous = cached.Snapshot;
            // The refreshed entry depends on everything the original materialization read plus
            // whatever the table re-evaluation read now, so it tracks the union.
            var mergedUserData = new SortedSet<int>(cached.UserDataIndices);
            recorder.MergeUserDataInto(mergedUserData);
            var userDataIndices = new int[mergedUserData.Count];
            mergedUserData.CopyTo(userDataIndices);
            var userDataWords = new uint[userDataIndices.Length];
            for (var index = 0; index < userDataIndices.Length; index++)
                userDataWords[index] = inputs.UserData[userDataIndices[index]];
            refreshed = new Entry
            {
                Plan = cached.Plan,
                UserDataIndices = userDataIndices,
                UserDataWords = userDataWords,
                ShaderBase = cached.ShaderBase,
                ComputeState = cached.ComputeState,
                RangeAddresses = cached.RangeAddresses,
                RangeOffsets = cached.RangeOffsets,
                RangeLengths = cached.RangeLengths,
                RangeClean = cached.RangeClean,
                WordTableOnly = cached.WordTableOnly,
                TableRefreshable = true,
                Bytes = current,
                Snapshot = new ResourceSnapshot
                {
                    Buffers = previous.Buffers,
                    Images = previous.Images,
                    Samplers = previous.Samplers,
                    FlattenedResourceTable = table,
                    UserData = previous.UserData,
                    DeviceAddressRanges = previous.DeviceAddressRanges,
                },
                Specialization = cached.Specialization,
                Next = cached.Next,
            };
            Store(key, refreshed);
            return true;
        }
        finally
        {
            ReturnRecorder(recorder);
        }
    }

    private ReadRecorder TakeRecorder()
    {
        var recorder = _spareRecorder ?? new ReadRecorder();
        _spareRecorder = null;
        return recorder;
    }

    private void ReturnRecorder(ReadRecorder recorder)
    {
        recorder.Reset();
        _spareRecorder = recorder;
    }

    // The byte offset of a recorded dword in the entry's bytes, found by its address.
    private static bool TryFindWord(Entry entry, ulong address, out int offset)
    {
        offset = 0;
        var addresses = entry.RangeAddresses;
        int low = 0, high = addresses.Length - 1, found = -1;
        while (low <= high)
        {
            var middle = (low + high) >>> 1;
            if (addresses[middle] <= address)
            {
                found = middle;
                low = middle + 1;
            }
            else
            {
                high = middle - 1;
            }
        }

        if (found < 0)
            return false;
        var delta = address - addresses[found];
        if (delta % sizeof(uint) != 0 || delta >= (ulong)entry.RangeLengths[found])
            return false;
        offset = entry.RangeOffsets[found] + (int)delta;
        return true;
    }

    private static ulong KeyOf(ShaderResourcePlan plan, ResourceRuntimeInputs inputs)
    {
        // User data is deliberately not hashed: it varies per draw, and an entry records which of
        // its words the plan actually read.
        var planHash = RuntimeHelpers.GetHashCode(plan);
        var hash = new HashCode();
        hash.Add(planHash);
        hash.Add(inputs.ShaderBase);
        hash.Add(inputs.ComputeState);
        var low = (uint)hash.ToHashCode();
        // A second, independent mix of the same identity keeps accidental collisions out of the
        // 64-bit key, which the user data no longer contributes to.
        var high = 0x9E3779B9u;
        high = (high ^ (uint)planHash) * 0x01000193u;
        high = (high ^ (uint)(inputs.ShaderBase >> 32)) * 0x01000193u;
        high = (high ^ (uint)inputs.ShaderBase) * 0x01000193u;
        high = (high ^ (uint)(inputs.ComputeState?.GetHashCode() ?? 0)) * 0x01000193u;
        return ((ulong)high << 32) | low;
    }

    // A hit reuses the descriptors, the flattened table and the specialization, but the user
    // data must be the current draw's: both backends build shaderData from it, so handing back
    // the entry's copy would push the earlier draw's push constants. When every word still
    // matches, the entry's snapshot is already this draw's and is handed back unchanged.
    private static ResourceSnapshot Rebind(ResourceSnapshot cached, ResourceRuntimeInputs inputs)
    {
        var cachedUserData = cached.UserData;
        var userData = inputs.UserData;
        if (cachedUserData.Length == userData.Count)
        {
            var identical = true;
            for (var index = 0; index < cachedUserData.Length; index++)
            {
                if (cachedUserData[index] != userData[index])
                {
                    identical = false;
                    break;
                }
            }

            if (identical)
                return cached;
        }

        return new ResourceSnapshot
        {
            Buffers = cached.Buffers,
            Images = cached.Images,
            Samplers = cached.Samplers,
            FlattenedResourceTable = cached.FlattenedResourceTable,
            UserData = userData as uint[] ?? userData.ToArray(),
            DeviceAddressRanges = cached.DeviceAddressRanges,
        };
    }

    // True when the key itself is resident, whether or not any variant matches. User data is
    // deliberately not part of the key, so a non-matching variant is the common case and must
    // extend the chain rather than replace it.
    private bool TryFind(ulong key, out Entry chain)
    {
        if (_young.TryGetValue(key, out chain!))
            return true;
        if (!_old.TryGetValue(key, out chain!))
            return false;
        _old.Remove(key);
        Store(key, chain);
        return true;
    }

    private void Store(ulong key, Entry entry)
    {
        if (_young.Count >= _generationCapacity && !_young.ContainsKey(key))
        {
            _old = _young;
            _young = new Dictionary<ulong, Entry>(_generationCapacity);
        }

        _young[key] = entry;
    }

    private bool Validate(Entry entry, ResidentGuestBytesReader residentReader, out bool unreadable)
    {
        unreadable = false;
        for (var index = 0; index < entry.RangeAddresses.Length; index++)
        {
            var length = entry.RangeLengths[index];
            if (_scratch.Length < length)
                _scratch = new byte[Math.Max(length, _scratch.Length * 2)];
            var current = _scratch.AsSpan(0, length);
            if (!residentReader(entry.RangeAddresses[index], current, entry.RangeClean[index]))
            {
                unreadable = true;
                return false;
            }

            if (!current.SequenceEqual(entry.Bytes.AsSpan(entry.RangeOffsets[index], length)))
                return false;
        }

        return true;
    }

    private sealed class Entry
    {
        public required ShaderResourcePlan Plan { get; init; }
        // The user-data words this entry's materialization actually read, ascending by index.
        // A draw whose other push constants changed still matches while these hold.
        public required int[] UserDataIndices { get; init; }
        public required uint[] UserDataWords { get; init; }
        public required ulong ShaderBase { get; init; }
        public required ComputeSelectorState? ComputeState { get; init; }
        public required ulong[] RangeAddresses { get; init; }
        public required int[] RangeOffsets { get; init; }
        public required int[] RangeLengths { get; init; }
        public required bool[] RangeClean { get; init; }
        // Per recorded dword: read only while the flattened table was evaluated.
        public required bool[] WordTableOnly { get; init; }
        // The table has the plan's plain layout, so no specialization read or extended it.
        public required bool TableRefreshable { get; init; }
        public required byte[] Bytes { get; init; }
        public required ResourceSnapshot Snapshot { get; init; }
        public required ResourceSpecialization Specialization { get; init; }
        public Entry? Next { get; set; }

        public bool Matches(ShaderResourcePlan plan, ResourceRuntimeInputs inputs)
        {
            if (!ReferenceEquals(Plan, plan) || ShaderBase != inputs.ShaderBase ||
                !Nullable.Equals(ComputeState, inputs.ComputeState))
                return false;
            return UserDataHolds(inputs);
        }

        // Two entries are interchangeable only when they captured the same guest words and depend on
        // the same push-data words. Comparing the push data alone would call every entry
        // identical for a plan that reads no user data, and evict a good entry on every miss.
        public bool SameSignature(Entry other)
        {
            if (!Bytes.AsSpan().SequenceEqual(other.Bytes.AsSpan()))
                return false;
            if (UserDataIndices.Length != other.UserDataIndices.Length)
                return false;
            for (var index = 0; index < UserDataIndices.Length; index++)
            {
                if (UserDataIndices[index] != other.UserDataIndices[index] ||
                    UserDataWords[index] != other.UserDataWords[index])
                    return false;
            }

            return true;
        }

        // Only the push-data words this entry depends on must still equal the current draw's;
        // the rest may differ without invalidating the descriptors it recorded.
        public bool UserDataHolds(ResourceRuntimeInputs inputs)
        {
            var userData = inputs.UserData;
            for (var index = 0; index < UserDataIndices.Length; index++)
            {
                var position = UserDataIndices[index];
                if (position >= userData.Count || userData[position] != UserDataWords[index])
                    return false;
            }

            return true;
        }
    }

    private sealed class ReadRecorder
    {
        private readonly List<(ulong Address, uint Word, bool Clean, bool Table)> _reads = new();
        private readonly SortedSet<int> _userData = new();
        private bool _inTable;
        private GuestWordReader? _reader;
        private GuestWordReader? _cleanReader;
        private ResidentGuestBytesReader? _residentReader;
        private readonly GuestWordReader _recordRead;
        private readonly GuestWordReader _recordCleanRead;
        private readonly ResidentGuestBytesReader _recordResidentRead;

        public ReadRecorder()
        {
            _recordRead = (ulong address, out uint word) => Read(_reader!, address, out word, clean: false);
            _recordCleanRead = (ulong address, out uint word) => Read(_cleanReader!, address, out word, clean: true);
            _recordResidentRead = (ulong address, Span<byte> destination, bool clean) => ReadResident(address, destination, clean);
            TablePhase = inTable =>
            {
                _inTable = inTable;
                ReadingTable = inTable;
            };
        }

        public void Reset()
        {
            _reads.Clear();
            _userData.Clear();
            // Retain ordinary descriptor walks without keeping unusually large
            // tables alive for the lifetime of the renderer.
            if (_reads.Capacity > 16384)
                _reads.Capacity = 0;
            _reader = null;
            _cleanReader = null;
            _residentReader = null;
            _inTable = false;
            Failed = false;
        }

        public bool Failed { get; private set; }

        public List<(ulong Address, uint Word, bool Clean, bool Table)> Reads => _reads;

        public Action<bool> TablePhase { get; }

        // Records that the materialization read user data at this index, so the entry only
        // invalidates when that particular word changes.
        public void RecordUserData(int index) => _userData.Add(index);

        public void MergeUserDataInto(SortedSet<int> target) => target.UnionWith(_userData);

        public GuestWordReader? Wrap(GuestWordReader? inner, bool clean)
        {
            if (inner is null)
                return null;
            if (clean)
            {
                _cleanReader = inner;
                return _recordCleanRead;
            }
            _reader = inner;
            return _recordRead;
        }

        private bool Read(GuestWordReader inner, ulong address, out uint word, bool clean)
        {
            if (!inner(address, out word))
            {
                Failed = true;
                return false;
            }
            _reads.Add((address, word, clean, _inTable));
            return true;
        }

        public ResidentGuestBytesReader? WrapResident(ResidentGuestBytesReader? inner)
        {
            if (inner is null)
                return null;
            _residentReader = inner;
            return _recordResidentRead;
        }

        private bool ReadResident(ulong address, Span<byte> destination, bool clean)
        {
            if (!_residentReader!(address, destination, clean))
                return false;

            for (var offset = 0; offset + sizeof(uint) <= destination.Length; offset += sizeof(uint))
                _reads.Add((address + (ulong)offset,
                    System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(destination[offset..]), clean, _inTable));
            return true;
        }

        public Entry Build(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, ResourceSnapshot snapshot,
            ResourceSpecialization specialization)
        {
            _reads.Sort((left, right) => left.Address.CompareTo(right.Address));
            // Count unique words and contiguous ranges first, then fill the
            // retained arrays directly. Temporary lists previously duplicated
            // every recorded byte and range on each cache miss.
            var rangeCount = 0;
            var wordCount = 0;
            ulong end = 0;
            var previous = ulong.MaxValue;
            foreach (var read in _reads)
            {
                if (read.Address == previous) continue;
                if (wordCount == 0 || read.Address != end) rangeCount++;
                wordCount++;
                previous = read.Address;
                end = read.Address + sizeof(uint);
            }

            var addresses = new ulong[rangeCount];
            var offsets = new int[rangeCount];
            var lengths = new int[rangeCount];
            var clean = new bool[rangeCount];
            var bytes = new byte[wordCount * sizeof(uint)];
            var tableOnly = new bool[wordCount];
            var rangeIndex = -1;
            var wordIndex = -1;
            end = 0;
            previous = ulong.MaxValue;
            foreach (var (address, word, wordClean, table) in _reads)
            {
                if (address == previous)
                {
                    clean[rangeIndex] |= wordClean;
                    tableOnly[wordIndex] &= table;
                    continue;
                }

                wordIndex++;
                if (rangeIndex < 0 || address != end)
                {
                    rangeIndex++;
                    addresses[rangeIndex] = address;
                    offsets[rangeIndex] = wordIndex * sizeof(uint);
                }
                lengths[rangeIndex] += sizeof(uint);
                clean[rangeIndex] |= wordClean;
                tableOnly[wordIndex] = table;
                System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(
                    bytes.AsSpan(wordIndex * sizeof(uint)), word);
                previous = address;
                end = address + sizeof(uint);
            }

            var userDataIndices = new int[_userData.Count];
            _userData.CopyTo(userDataIndices);
            var userDataWords = new uint[userDataIndices.Length];
            for (var index = 0; index < userDataIndices.Length; index++)
                userDataWords[index] = inputs.UserData[userDataIndices[index]];
            return new Entry
            {
                Plan = plan,
                UserDataIndices = userDataIndices,
                UserDataWords = userDataWords,
                ShaderBase = inputs.ShaderBase,
                ComputeState = inputs.ComputeState,
                RangeAddresses = addresses,
                RangeOffsets = offsets,
                RangeLengths = lengths,
                RangeClean = clean,
                WordTableOnly = tableOnly,
                TableRefreshable = snapshot.FlattenedResourceTable.Length ==
                    plan.TableReads.Count + plan.WrittenRangeCount * ShaderResourcePlan.WrittenRangeDwordCount,
                Bytes = bytes,
                Snapshot = snapshot,
                Specialization = specialization,
            };
        }
    }
}
