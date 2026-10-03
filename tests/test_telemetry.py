import ctypes
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_llm.telemetry import SystemTelemetry, linux_temperature, parse_vm_stat, macos_memory_pressure
from local_llm.sensors_macos import KeyData, MacSensors


class TelemetryTests(unittest.TestCase):
    def test_pressure_reads_dispatch_masks_and_never_guesses_on_failure(self):
        with patch('local_llm.telemetry.platform.system', return_value='Darwin'):
            for raw, expected in [('1','normal'),('2','warning'),('4','critical'),('0',None),('bad',None)]:
                with patch('local_llm.telemetry.subprocess.check_output',return_value=raw):
                    self.assertEqual(macos_memory_pressure(),expected)
            with patch('local_llm.telemetry.subprocess.check_output',side_effect=OSError):
                self.assertIsNone(macos_memory_pressure())
        with patch('local_llm.telemetry.platform.system', return_value='Linux'), patch('local_llm.telemetry.subprocess.check_output') as command:
            self.assertIsNone(macos_memory_pressure()); command.assert_not_called()

    def test_forced_snapshot_refreshes_even_inside_poll_cache_window(self):
        with patch('local_llm.telemetry.platform.system', return_value='Darwin'), patch('local_llm.telemetry.detect_hardware', return_value={'memory_bytes':24*1024**3}), patch('local_llm.sensors_macos.MacSensors',side_effect=OSError), patch('local_llm.telemetry.subprocess.check_output',side_effect=OSError) as command:
            telemetry=SystemTelemetry()
            telemetry.snapshot();telemetry.snapshot();telemetry.snapshot(refresh=True)
            self.assertEqual(command.call_count,2)

    def test_mac_memory_excludes_reclaimable_files_and_purgeable_pages(self):
        text = '''Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages active: 100.
Pages inactive: 50.
Pages wired down: 20.
Pages speculative: 5.
Pages occupied by compressor: 10.
Pages purgeable: 5.
File-backed pages: 30.
'''
        self.assertEqual(parse_vm_stat(text, 10000000), 150 * 16384)
        self.assertEqual(parse_vm_stat(text, 10), 10)
        with self.assertRaises(ValueError):
            parse_vm_stat('unavailable', 10000000)

    def test_linux_reads_cpu_sensors_in_celsius_and_excludes_gpu_or_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for index, name, temperature in [(0, 'coretemp', '48000'), (1, 'k10temp', 'nan'), (2, 'amdgpu', '80000')]:
                path = root / ('hwmon' + str(index)); path.mkdir()
                (path / 'name').write_text(name); (path / 'temp1_input').write_text(temperature)
            self.assertEqual(linux_temperature(root), 48)
            (root / 'hwmon0/temp1_input').write_text('0')
            self.assertIsNone(linux_temperature(root))

    def test_smc_temperature_uses_actual_values_and_discards_invalid_readings(self):
        self.assertEqual(ctypes.sizeof(KeyData), 80)
        sensor = MacSensors.__new__(MacSensors)
        sensor.keys = {'Tp00': 'flt', 'Te00': 'sp78', 'Ts00': 'flt'}
        samples = {'Tp00': (struct.pack('<f', 40), b'flt '), 'Te00': ((60 * 256).to_bytes(2, 'big'), b'sp78'), 'Ts00': (struct.pack('<f', float('nan')), b'flt ')}
        sensor.read = lambda key: samples[key]
        self.assertEqual(sensor.temperature(), 50)
        samples['Tp00'] = (struct.pack('<f', 0), b'flt ')
        samples['Te00'] = (b'\x00\x00', b'sp78')
        self.assertIsNone(sensor.temperature())

    def test_snapshot_cache_is_bounded_and_unknown_metrics_remain_unknown(self):
        with patch('local_llm.telemetry.platform.system', return_value='Darwin'), patch('local_llm.telemetry.detect_hardware', return_value={'memory_bytes': 24 * 1024 ** 3}), patch('local_llm.sensors_macos.MacSensors', side_effect=OSError), patch('local_llm.telemetry.subprocess.check_output', side_effect=OSError) as command, patch('local_llm.telemetry.time.monotonic', return_value=1):
            telemetry = SystemTelemetry(); first = telemetry.snapshot(); second = telemetry.snapshot()
            self.assertEqual(command.call_count, 1)
            self.assertEqual(first, second)
            second['memory_used_bytes'] = 123
            self.assertIsNone(telemetry.snapshot()['memory_used_bytes'])
            self.assertIsNone(first['cpu_temperature_celsius'])
            self.assertIsNone(first['process_rss_bytes'])
            json.dumps(first, allow_nan=False)
