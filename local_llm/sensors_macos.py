"""Read-only Apple SMC temperature access, with no helper or administrator rights.

The SMC wire layout and Tp/Te/Ts sensor convention can also be inspected in
https://github.com/vladkens/macmon/blob/main/src_lib/sources.rs.
These sensor names are not a public Apple contract; unavailable readings stay null.
"""
import ctypes as c
import math
import struct


class Version(c.Structure):
    _fields_ = [(k, c.c_uint8) for k in ('major', 'minor', 'build', 'reserved')] + [('release', c.c_uint16)]


class Limits(c.Structure):
    _fields_ = [('version', c.c_uint16), ('length', c.c_uint16)] + [(k, c.c_uint32) for k in ('cpu', 'gpu', 'memory')]


class KeyInfo(c.Structure):
    _fields_ = [('size', c.c_uint32), ('type', c.c_uint32), ('attributes', c.c_uint8)]


class KeyData(c.Structure):
    _fields_ = [('key', c.c_uint32), ('version', Version), ('limits', Limits),
                ('info', KeyInfo), ('result', c.c_uint8), ('status', c.c_uint8),
                ('command', c.c_uint8), ('index', c.c_uint32), ('data', c.c_uint8 * 32)]


class MacSensors:
    def __init__(self):
        self.io = c.CDLL('/System/Library/Frameworks/IOKit.framework/IOKit')
        self.lib = c.CDLL('/usr/lib/libSystem.B.dylib')
        signatures = {
            'IOServiceMatching': ([c.c_char_p], c.c_void_p),
            'IOServiceGetMatchingServices': ([c.c_uint32, c.c_void_p, c.POINTER(c.c_uint32)], c.c_int),
            'IOIteratorNext': ([c.c_uint32], c.c_uint32),
            'IORegistryEntryGetName': ([c.c_uint32, c.c_void_p], c.c_int),
            'IOObjectRelease': ([c.c_uint32], c.c_int),
            'IOServiceOpen': ([c.c_uint32, c.c_uint32, c.c_uint32, c.POINTER(c.c_uint32)], c.c_int),
            'IOServiceClose': ([c.c_uint32], c.c_int),
            'IOConnectCallStructMethod': ([c.c_uint32, c.c_uint32, c.c_void_p, c.c_size_t, c.c_void_p, c.POINTER(c.c_size_t)], c.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.io, name)
            fn.argtypes = args
            fn.restype = result
        self.lib.mach_task_self.restype = c.c_uint32
        self.connection = c.c_uint32()
        iterator = c.c_uint32()
        matching = self.io.IOServiceMatching(b'AppleSMC')
        if not matching or self.io.IOServiceGetMatchingServices(0, matching, c.byref(iterator)):
            raise OSError('Apple SMC indisponible')
        try:
            while True:
                service = self.io.IOIteratorNext(iterator)
                if not service:
                    break
                try:
                    name = c.create_string_buffer(128)
                    if self.io.IORegistryEntryGetName(service, name):
                        continue
                    if name.value not in {b'AppleSMCKeysEndpoint', b'AppleSMC'}:
                        continue
                    if not self.io.IOServiceOpen(service, self.lib.mach_task_self(), 0, c.byref(self.connection)):
                        break
                finally:
                    self.io.IOObjectRelease(service)
        finally:
            self.io.IOObjectRelease(iterator)
        if not self.connection.value:
            raise OSError('Capteurs SMC non accessibles')
        self.keys = {}
        try:
            count = int.from_bytes(self.read('#KEY')[0][:4], 'big')
            for index in range(min(count, 8192)):
                try:
                    request = KeyData()
                    request.command = 8
                    request.index = index
                    name = self.call(request).key.to_bytes(4, 'big').decode('ascii')
                    if name.startswith(('Tp', 'Te', 'Ts', 'TC')):
                        data, kind = self.read(name)
                        if kind in {b'flt ', b'sp78'}:
                            self.keys[name] = kind
                except (OSError, ValueError, UnicodeError):
                    continue
        except Exception:
            self.close()
            raise

    def call(self, request):
        output = KeyData()
        size = c.c_size_t(c.sizeof(output))
        code = self.io.IOConnectCallStructMethod(
            self.connection, 2, c.byref(request), c.sizeof(request),
            c.byref(output), c.byref(size),
        )
        if code or output.result or size.value != c.sizeof(output):
            raise OSError('Lecture SMC indisponible')
        return output

    def read(self, name):
        request = KeyData()
        request.key = int.from_bytes(name.encode('ascii'), 'big')
        request.command = 9
        info = self.call(request).info
        if not 0 < info.size <= 32:
            raise ValueError('Taille SMC invalide')
        request.info = info
        request.command = 5
        return bytes(self.call(request).data[:info.size]), info.type.to_bytes(4, 'big')

    def temperature(self):
        values = []
        for key in self.keys:
            try:
                data, kind = self.read(key)
                if kind == b'flt ' and len(data) == 4:
                    value = struct.unpack('<f', data)[0]
                elif kind == b'sp78' and len(data) == 2:
                    value = int.from_bytes(data, 'big', signed=True) / 256
                else:
                    continue
                if value is not None and math.isfinite(value) and 0 < value <= 150:
                    values.append(value)
            except (OSError, ValueError, struct.error):
                continue
        return sum(values) / len(values) if values else None

    def close(self):
        if self.connection.value:
            self.io.IOServiceClose(self.connection)
            self.connection.value = 0
