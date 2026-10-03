// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.Diagnostics;

internal static class GpuReadTrace
{
    private const int MaxReports = 64;

    public static readonly bool Enabled =
        Environment.GetEnvironmentVariable("SHARPEMU_TRACE_GPU_READS") == "1";

    [ThreadStatic]
    public static ulong CurrentShader;

    [ThreadStatic]
    public static string? CurrentStage;

    private static readonly HashSet<(ulong Shader, ulong Page, bool Table)> _reported = new();

    public static void Record(ulong address, bool table)
    {
        var key = (CurrentShader, address & ~0xFFFUL, table);
        lock (_reported)
        {
            if (_reported.Count >= MaxReports || !_reported.Add(key))
            {
                return;
            }
        }

        Console.Error.WriteLine(
            $"[GPU][SYNC_READ] shader=0x{CurrentShader:X16} stage={CurrentStage} address=0x{address:X16} table={table}");
    }
}
