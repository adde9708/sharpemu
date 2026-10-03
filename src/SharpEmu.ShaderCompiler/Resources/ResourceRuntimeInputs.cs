// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;

namespace SharpEmu.ShaderCompiler.Resources;

// Reads one dword of guest memory; false when the address cannot be read.
public delegate bool GuestWordReader(ulong address, out uint word);

// What one draw supplies to materialise a plan: its user data, the shader base and
// the two memory readers. The clean reader refuses memory the GPU may still own.
public sealed class ResourceRuntimeInputs
{
    public IReadOnlyList<uint> UserData { get; init; } = [];
    public ulong ShaderBase { get; init; }
    public GuestWordReader? ReadMemory { get; init; }
    public GuestWordReader? ReadCleanMemory { get; init; }
    public ComputeSelectorState? ComputeState { get; init; }

    // Told true before the flattened table's words are evaluated and false after, so a reader
    // wrapper can tell the words only the table reads from those the descriptors depend on.
    public Action<bool>? TablePhase { get; init; }

// Told with the index of every user-data word the plan actually evaluates. User data is read
    // straight out of UserData rather than through a word reader, so a reader wrapper cannot see
    // it; this is the only place that consumption can be observed.
    public Action<int>? UserDataRead { get; init; }
    public ResidentGuestBytesReader? ReadResidentMemory { get; init; }

    public bool ReadsClean { get; init; }

    public ResourceRuntimeInputs WithReader(GuestWordReader? reader) => new()
    {
        UserData = UserData,
        ShaderBase = ShaderBase,
        ReadMemory = reader,
        ReadCleanMemory = ReadCleanMemory,
        ComputeState = ComputeState,
        TablePhase = TablePhase,
UserDataRead = UserDataRead,
        ReadResidentMemory = ReadResidentMemory,
        ReadsClean = ReadsClean || ReferenceEquals(reader, ReadCleanMemory),
    };
}

public readonly record struct ComputeSelectorState(uint WaveSize, uint ThreadsX, uint ThreadsY, uint ThreadsZ,
    bool HasPartialWorkgroups, uint LocalDataShareDwords, int LocalInvocationIdComponents);
