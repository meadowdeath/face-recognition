"""Real QtDBus wire contract against a private, synthetic GIO service on Linux.

Optional fixture uses existing system GIO/dbus-daemon only; it never connects to
the system bus and does not require installing these tools for portable tests.
"""
import os
import shutil
import subprocess
import sys
import unittest

SERVER = r'''
import sys
from gi.repository import Gio, GLib
c=Gio.DBusConnection.new_for_address_sync(sys.argv[1],Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT|Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,None,None)
c.call_sync('org.freedesktop.DBus','/org/freedesktop/DBus','org.freedesktop.DBus','RequestName',GLib.Variant('(su)',('org.freedesktop.UDisks2',0)),None,0,5000,None)
xml=''' + '"""' + '''<node><interface name="org.freedesktop.DBus.ObjectManager"><method name="GetManagedObjects"><arg type="a{oa{sa{sv}}}" direction="out"/></method></interface><interface name="org.freedesktop.UDisks2.Filesystem"><method name="Mount"><arg type="a{sv}" direction="in"/><arg type="s" direction="out"/></method></interface></node>''' + '"""' + r'''
info=Gio.DBusNodeInfo.new_for_xml(xml)
def call(conn,sender,path,interface,method,args,invocation):
    if method=='GetManagedObjects':
        data={'/org/freedesktop/UDisks2/block_devices/synthetic':{'org.freedesktop.UDisks2.Block':{'Drive':GLib.Variant('o','/org/freedesktop/UDisks2/drives/synthetic')},'org.freedesktop.UDisks2.Filesystem':{'MountPoints':GLib.Variant('aay',[list('/media/Prueba Á\0'.encode())])}}}
        invocation.return_value(GLib.Variant('(a{oa{sa{sv}}})',(data,)))
    elif args.get_type_string()=='(a{sv})' and args.unpack()[0]=={'auth.no_user_interaction':True}:
        invocation.return_value(GLib.Variant('(s)',('/media/Prueba Á',)))
    else:invocation.return_dbus_error('org.freedesktop.DBus.Error.InvalidArgs','Invalid options signature')
for i in info.interfaces:c.register_object('/org/freedesktop/UDisks2',i,call,None,None)
print('READY',flush=True)
GLib.MainLoop().run()
'''


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('dbus-daemon'), 'Private Linux D-Bus fixture unavailable')
class QtDBusContractTests(unittest.TestCase):
    def test_nested_managed_objects_byte_arrays_object_paths_and_typed_options_over_wire(self):
        interpreter = '/usr/bin/python3'
        if not os.path.exists(interpreter) or subprocess.run([interpreter, '-c', 'from gi.repository import Gio'], capture_output=True).returncode:
            self.skipTest('Optional system GIO fixture unavailable')
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtDBus import QDBusConnection
        from face_recognition.infrastructure.udisks import UDisksProvider, ROOT, FS, BLOCK, options, mount_path
        app = QApplication.instance() or QApplication([])
        daemon = subprocess.Popen(['dbus-daemon', '--session', '--nofork', '--print-address=1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        service = None; name = 'faceid-private-contract'
        try:
            address = daemon.stdout.readline().strip()
            service = subprocess.Popen([interpreter, '-u', '-c', SERVER, address], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            import select
            if not select.select([service.stdout], [], [], 10)[0]: self.fail('Synthetic D-Bus service startup timed out')
            self.assertEqual(service.stdout.readline().strip(), 'READY')
            bus = QDBusConnection.connectToBus(address, name)
            self.assertTrue(bus.isConnected())
            provider = UDisksProvider([], bus=bus)
            objects = provider.objects()
            block = objects['/org/freedesktop/UDisks2/block_devices/synthetic']
            self.assertEqual(block[BLOCK]['Drive'], '/org/freedesktop/UDisks2/drives/synthetic')
            self.assertEqual(mount_path(block[FS]['MountPoints'][0]), '/media/Prueba Á')
            self.assertEqual(provider.call(ROOT, FS, 'Mount', [options()]), ['/media/Prueba Á'])
        finally:
            QDBusConnection.disconnectFromBus(name)
            for process in (service, daemon):
                if process:
                    process.terminate()
                    try: process.communicate(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.communicate()
