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
public sealed class ResourceMaterializationCache(int generationCapacity = 16384)
{
    // One key covers a plan and its shader base; the entry list holds the recent user-data
    // signatures seen under it, so alternating draws keep their own entry instead of evicting
    // each other.
    private const int MaxSignaturesPerKey = 8;

    // Two generations approximate LRU: a full young generation becomes the old one,
    // and an old entry that is used again is promoted back.
    private readonly int _generationCapacity = Math.Max(1, generationCapacity);
    private Dictionary<ulong, List<Entry>> _young = new();
    private Dictionary<ulong, List<Entry>> _old = new();
    private byte[] _scratch = new byte[256];
    // Full-entry layout: every recorded range written at its own offset, so a rejected
    // validation is already in the shape the table refresh needs and the guest is not read twice.
    private byte[] _current = new byte[256];

    private static long _totalHits;
    private static long _totalMisses;
    private static long _totalUncacheable;
    private static long _totalTableRefreshes;

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
        var refreshes = Interlocked.Exchange(ref _totalTableRefreshes, 0);
        var total = hits + misses;
        return FormattableString.Invariant(
            $"[PERF][RESOURCE_CACHE] hits={hits} misses={misses} uncacheable={uncacheable} refreshes={refreshes} success_rate={(total == 0 ? 0 : (hits + refreshes) * 100.0 / (total + refreshes)):F1}%");
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
        if (TryFind(key, plan, inputs, out var cached))
        {
            if (Validate(cached, residentReader, cached.TableRefreshable, out var captured))
            {
                Hits++;
                Interlocked.Increment(ref _totalHits);
                snapshot = Rebind(cached.Snapshot, inputs);
                specialization = cached.Specialization;
                failure = default;
                return true;
            }

            if (TryRefreshTable(key, cached, plan, inputs, residentReader, captured, out var refreshed))
            {
                TableRefreshes++;
                Interlocked.Increment(ref _totalTableRefreshes);
                snapshot = Rebind(refreshed.Snapshot, inputs);
                specialization = refreshed.Specialization;
                failure = default;
                return true;
            }
        }

        Misses++;
        Interlocked.Increment(ref _totalMisses);
        var recorder = new ReadRecorder();
        var recording = new ResourceRuntimeInputs
        {
            UserData = inputs.UserData,
            ShaderBase = inputs.ShaderBase,
            ReadMemory = recorder.Wrap(inputs.ReadMemory, clean: false),
            ReadCleanMemory = recorder.Wrap(inputs.ReadCleanMemory, clean: true),
            ComputeState = inputs.ComputeState,
            TablePhase = recorder.SetTablePhase,
            UserDataRead = recorder.RecordUserData,
        };
        if (!ResourceMaterializer.Materialize(plan, recording, ref snapshot, ref specialization, out failure))
            return false;

        if (recorder.Failed)
        {
            Uncacheable++;
            Interlocked.Increment(ref _totalUncacheable);
            return true;
        }

        Store(key, recorder.Build(plan, inputs, snapshot, specialization));
        return true;
    }

    // Most stale entries in Demon's Souls differ only in words the shader reads through scalar
    // loads (inline constants rewritten every frame); the descriptors, device ranges and
    // specialization are unchanged. When every changed word was read only while evaluating the
    // flattened table, only the table is evaluated again. It is refused when the plan's
    // specialization reads or extends the table (indirect or candidate tables), when the table
    // now reads a word the entry did not validate, or when a word moved between the two reads.
    private bool TryRefreshTable(ulong key, Entry cached, ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        ResidentGuestBytesReader residentReader, bool captured, out Entry refreshed)
    {
        refreshed = null!;
        if (!cached.TableRefreshable)
            return false;

        // The refresh only re-evaluates the flattened table, so it is defined for a draw whose
        // user data still matches; a draw that changed one of these words needs a full walk.
        if (!cached.UserDataHolds(inputs))
            return false;

        if (_current.Length < cached.Bytes.Length)
            _current = new byte[Math.Max(cached.Bytes.Length, _current.Length * 2)];

        // Validate already read every range into _current at its recorded offset, so the
        // common rejected-validation path costs no second guest read. A read that failed
        // leaves captured false and the ranges are read again here.
        if (!captured)
        {
            for (var index = 0; index < cached.RangeAddresses.Length; index++)
            {
                if (!residentReader(cached.RangeAddresses[index],
                        _current.AsSpan(cached.RangeOffsets[index], cached.RangeLengths[index]), cached.RangeClean[index]))
                    return false;
            }
        }

        var current = _current.AsSpan(0, cached.Bytes.Length);

        var changed = false;
        for (var offset = 0; offset < current.Length; offset += sizeof(uint))
        {
            if (current.Slice(offset, sizeof(uint)).SequenceEqual(cached.Bytes.AsSpan(offset, sizeof(uint))))
                continue;
            if (!cached.WordTableOnly[offset / sizeof(uint)])
                return false;
            changed = true;
        }

        if (!changed)
            return false;

        var recorder = new ReadRecorder();
        var recording = new ResourceRuntimeInputs
        {
            UserData = inputs.UserData,
            ShaderBase = inputs.ShaderBase,
            ReadMemory = recorder.Wrap(inputs.ReadMemory, clean: false),
            ReadCleanMemory = recorder.Wrap(inputs.ReadCleanMemory, clean: true),
            ComputeState = inputs.ComputeState,
        };
        var cachedTable = cached.Snapshot.FlattenedResourceTable;
        if (!ResourceMaterializer.TryEvaluateTable(plan, recording, out var table) || recorder.Failed || table.Length != cachedTable.Length)
            return false;

        foreach (var (address, word, _, _) in recorder.Reads)
        {
            if (!TryFindWord(cached, address, out var offset) ||
                System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(current.Slice(offset, sizeof(uint))) != word)
                return false;
        }

        // The written device-address slots come from reads validated unchanged above.
        foreach (var slot in plan.WrittenRangeSlotByHandle.Values)
            cachedTable.AsSpan((int)slot, ShaderResourcePlan.WrittenRangeDwordCount).CopyTo(table.AsSpan((int)slot));

        var previous = cached.Snapshot;
        refreshed = new Entry
        {
            Plan = cached.Plan,
            UserDataIndices = cached.UserDataIndices,
            UserDataWords = cached.UserDataWords,
            ShaderBase = cached.ShaderBase,
            ComputeState = cached.ComputeState,
            RangeAddresses = cached.RangeAddresses,
            RangeOffsets = cached.RangeOffsets,
            RangeLengths = cached.RangeLengths,
            RangeClean = cached.RangeClean,
            WordTableOnly = cached.WordTableOnly,
            TableRefreshable = true,
            // Copied, not aliased: the entry outlives this call and _current is reused.
            Bytes = current.ToArray(),
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
        };
        Store(key, refreshed);
        return true;
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
        // its words the plan read.
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

    // A hit reuses the descriptor words, the flattened table and the specialization, but the
    // user data is the current draw's: both backends build the shaderData block from it, so
    // handing back the entry's copy would push the earlier draw's push constants. When every
    // word still matches, the entry's snapshot is already that draw's and is handed back as is.
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

    private bool TryFind(ulong key, ShaderResourcePlan plan, ResourceRuntimeInputs inputs, out Entry entry)
    {
        entry = null!;
        if (!_young.TryGetValue(key, out var young))
            return Promote(key, plan, inputs, out entry);

        var found = IndexOfMatching(young, plan, inputs);
        if (found < 0)
            return Promote(key, plan, inputs, out entry);

        entry = young[found];
        young.RemoveAt(found);
        young.Insert(0, entry);
        return true;
    }

    private bool Promote(ulong key, ShaderResourcePlan plan, ResourceRuntimeInputs inputs, out Entry entry)
    {
        entry = null!;
        if (!_old.TryGetValue(key, out var old))
            return false;

        var found = IndexOfMatching(old, plan, inputs);
        if (found < 0)
            return false;

        entry = old[found];
        old.RemoveAt(found);
        _old.Remove(key);
        Store(key, entry);
        return true;
    }

    private static int IndexOfMatching(List<Entry> entries, ShaderResourcePlan plan, ResourceRuntimeInputs inputs)
    {
        for (var index = 0; index < entries.Count; index++)
        {
            if (entries[index].Matches(plan, inputs))
                return index;
        }

        return -1;
    }

    private void Store(ulong key, Entry entry)
    {
        if (_young.Count >= _generationCapacity && !_young.ContainsKey(key))
        {
            _old = _young;
            _young = new Dictionary<ulong, List<Entry>>(_generationCapacity);
        }

        if (!_young.TryGetValue(key, out var entries))
        {
            entries = [];
            _young[key] = entries;
        }

        // A signature already held is replaced rather than duplicated, so a draw that
        // alternates between a few push-data blocks does not fill the list with copies.
        var existing = IndexOfSignature(entries, entry);
        if (existing >= 0)
            entries.RemoveAt(existing);
        entries.Insert(0, entry);
        if (entries.Count > MaxSignaturesPerKey)
            entries.RemoveAt(entries.Count - 1);
    }

    private static int IndexOfSignature(List<Entry> entries, Entry entry)
    {
        for (var index = 0; index < entries.Count; index++)
        {
            if (entries[index].SameSignature(entry))
                return index;
        }

        return -1;
    }

    private bool Validate(Entry entry, ResidentGuestBytesReader residentReader, bool capture, out bool captured)
    {
        captured = false;
        if (capture)
        {
            if (_current.Length < entry.Bytes.Length)
                _current = new byte[Math.Max(entry.Bytes.Length, _current.Length * 2)];
            for (var index = 0; index < entry.RangeAddresses.Length; index++)
            {
                if (!residentReader(entry.RangeAddresses[index],
                        _current.AsSpan(entry.RangeOffsets[index], entry.RangeLengths[index]), entry.RangeClean[index]))
                    return false;
            }

            for (var index = 0; index < entry.RangeAddresses.Length; index++)
                if (!_current.AsSpan(entry.RangeOffsets[index], entry.RangeLengths[index])
                        .SequenceEqual(entry.Bytes.AsSpan(entry.RangeOffsets[index], entry.RangeLengths[index])))
                    return false;

            captured = true;
            return true;
        }

        for (var index = 0; index < entry.RangeAddresses.Length; index++)
        {
            var length = entry.RangeLengths[index];
            if (_scratch.Length < length)
                _scratch = new byte[Math.Max(length, _scratch.Length * 2)];
            var current = _scratch.AsSpan(0, length);
            if (!residentReader(entry.RangeAddresses[index], current, entry.RangeClean[index]) ||
                !current.SequenceEqual(entry.Bytes.AsSpan(entry.RangeOffsets[index], length)))
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

        public bool Matches(ShaderResourcePlan plan, ResourceRuntimeInputs inputs)
        {
            if (!ReferenceEquals(Plan, plan) || ShaderBase != inputs.ShaderBase ||
                !Nullable.Equals(ComputeState, inputs.ComputeState))
                return false;
            return UserDataHolds(inputs);
        }

        // The same push-data words, whether or not they still equal the current draw's.
        public bool SameSignature(Entry other)
        {
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

        public bool Failed { get; private set; }

        public List<(ulong Address, uint Word, bool Clean, bool Table)> Reads => _reads;

        public void SetTablePhase(bool inTable) => _inTable = inTable;

        public void RecordUserData(int index) => _userData.Add(index);

        public GuestWordReader? Wrap(GuestWordReader? inner, bool clean)
        {
            if (inner is null)
                return null;
            return (ulong address, out uint word) =>
            {
                if (!inner(address, out word))
                {
                    Failed = true;
                    return false;
                }

                _reads.Add((address, word, clean, _inTable));
                return true;
            };
        }

        public Entry Build(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, ResourceSnapshot snapshot,
            ResourceSpecialization specialization)
        {
            _reads.Sort((left, right) => left.Address.CompareTo(right.Address));
            var addresses = new List<ulong>();
            var offsets = new List<int>();
            var lengths = new List<int>();
            var clean = new List<bool>();
            var bytes = new List<byte>(_reads.Count * sizeof(uint));
            var tableOnly = new List<bool>(_reads.Count);
            ulong end = 0;
            var previous = ulong.MaxValue;
            foreach (var (address, word, wordClean, table) in _reads)
            {
                if (address == previous)
                {
                    clean[^1] |= wordClean;
                    tableOnly[^1] &= table;
                    continue;
                }

                tableOnly.Add(table);

                previous = address;
                if (addresses.Count == 0 || address != end)
                {
                    addresses.Add(address);
                    offsets.Add(bytes.Count);
                    lengths.Add(0);
                    clean.Add(false);
                }

                lengths[^1] += sizeof(uint);
                clean[^1] |= wordClean;
                bytes.Add((byte)word);
                bytes.Add((byte)(word >> 8));
                bytes.Add((byte)(word >> 16));
                bytes.Add((byte)(word >> 24));
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
                RangeAddresses = [.. addresses],
                RangeOffsets = [.. offsets],
                RangeLengths = [.. lengths],
                RangeClean = [.. clean],
                WordTableOnly = [.. tableOnly],
                TableRefreshable = snapshot.FlattenedResourceTable.Length ==
                    plan.TableReads.Count + plan.WrittenRangeCount * ShaderResourcePlan.WrittenRangeDwordCount,
                Bytes = [.. bytes],
                Snapshot = snapshot,
                Specialization = specialization,
            };
        }
    }
}
