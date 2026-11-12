import struct
import unittest

from lan_protocol import (CAT_GIMBAL, CAT_MODE, CAT_PIPELINE, CAT_SYSTEM, ControlLink,
                          METADATA, TELEMETRY, TRACK, decode_metadata, encode)


class FakeSocket:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.sent = []
        self.closed = False

    def settimeout(self, _seconds):
        pass

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, length):
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) > length:
            self.chunks.insert(0, chunk[length:])
        return chunk[:length]

    def close(self):
        self.closed = True


class WireTest(unittest.TestCase):
    def test_gimbal_position_estimate(self):
        sock = FakeSocket([encode(CAT_GIMBAL, 0x84,
                                  struct.pack(">Bff", 1, 52.5, 40.0))])
        client = ControlLink("unused")
        client._socket = sock
        self.assertEqual(client.position(), dict(pan=52.5, tilt=40.0))
        self.assertEqual(sock.sent[0], encode(CAT_GIMBAL, 4))

    def test_metadata(self):
        packet = METADATA.pack(b"LWIR", 7, 9000, 123456, 1)
        packet += TRACK.pack(42, 10, 20, 30, 40, 0, 0.9)
        decoded = decode_metadata(packet)
        self.assertEqual((decoded["frame_id"], decoded["rtp_timestamp"]), (7, 9000))
        self.assertEqual(decoded["tracks"][0]["id"], 42)
        with self.assertRaises(Exception):
            decode_metadata(packet[:-1])
        footer = TELEMETRY.pack(b"LWTM", 1, 2, 3, 6, 1, 0.0, 1, 2,
                                *([1.5] + [0.0] * 17))
        decoded = decode_metadata(packet + footer)
        self.assertEqual(decoded["telemetry"]["values"][0], 1.5)
        with self.assertRaises(Exception):
            decode_metadata(packet + b"trailing")

    def test_catalog_change_waits_until_completed(self):
        status = encode(CAT_SYSTEM, 3, struct.pack(">BBBBI", 0, 0, 0, 0, 0))
        count = encode(CAT_PIPELINE, 0x81, struct.pack(">II", 1, 1))
        info = encode(CAT_PIPELINE, 0x85, struct.pack(">III", 2, 1, 1) +
                      b"NEWMRW480\0".ljust(64, b"\0"))
        accepted = encode(CAT_PIPELINE, 0x83, struct.pack(">IBB", 3, 1, 0))
        completed = encode(CAT_PIPELINE, 0x83, struct.pack(">IBB", 3, 2, 0))
        sock = FakeSocket([status[:3], status[3:], count, info[:7], info[7:],
                           accepted, completed])
        client = ControlLink("unused")
        client._socket = sock
        catalog = client.catalog("model")
        self.assertEqual(catalog, [dict(id=1, name="NEWMRW480", available=True,
                                        active=False)])
        self.assertEqual(client.change("model", 1), 0)
        self.assertEqual([frame[5] for frame in sock.sent], [1, 5, 3])
        self.assertNotIn("model", client._catalog)

    def test_manual_ack(self):
        sock = FakeSocket([encode(CAT_SYSTEM, 1, bytes((CAT_MODE, 2, 0)))])
        client = ControlLink("unused")
        client._socket = sock
        self.assertEqual(client.mode(None), 0)
        self.assertEqual(sock.sent[0], encode(CAT_MODE, 2))

    def test_tracker_policy(self):
        payload = struct.pack(">IIIIfIII", 1, 2, 8000, 64000, 0.5, 3, 130, 7)
        sock = FakeSocket([encode(CAT_PIPELINE, 0x87, payload)])
        client = ControlLink("unused")
        client._socket = sock
        policy = client.tracker_policy()
        self.assertEqual(policy["tracker_id"], 2)
        self.assertEqual(policy["confirmation_ms"], 130)
        self.assertTrue(policy["gmc"])
        self.assertTrue(policy["post_reid"])
        self.assertEqual(sock.sent[0], encode(CAT_PIPELINE, 7, struct.pack(">I", 1)))

    def test_tracker_policy_rejects_malformed_payload(self):
        sock = FakeSocket([encode(CAT_PIPELINE, 0x87, b"\0" * 31)])
        client = ControlLink("unused")
        client._socket = sock
        with self.assertRaises(Exception):
            client.tracker_policy()

    def test_change_policy_waits_for_completed_result(self):
        accepted = encode(CAT_PIPELINE, 0x88, struct.pack(">IBB", 1, 1, 0))
        completed = encode(CAT_PIPELINE, 0x88, struct.pack(">IBB", 1, 2, 0))
        sock = FakeSocket([accepted, completed])
        client = ControlLink("unused")
        client._socket = sock
        self.assertEqual(client.change_policy(warmth_min=8000, edge_birth=0.5,
                                              confirmation_ms=130), 0)
        self.assertEqual(sock.sent[0], encode(CAT_PIPELINE, 8,
            struct.pack(">IIIIfII", 1, 13, 8000, 0, 0.5, 130, 0)))
        with self.assertRaises(ValueError):
            client.change_policy(warmth_min=70000)


if __name__ == "__main__":
    unittest.main()
