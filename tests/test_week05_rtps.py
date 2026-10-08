"""The RTPS wire format: synthetic round trips and the committed captures of real ROS 2 nodes."""

from __future__ import annotations

from pathlib import Path

import pytest

from week05 import cdr, pcap, rtps
from week05.dissect import dissect, dissect_pcap

CAPTURES = Path(__file__).resolve().parents[1] / "docs" / "week05" / "captures"


def test_port_mapping():
    assert rtps.spdp_multicast_port(0) == 7400
    assert rtps.metatraffic_unicast_port(0, 1) == 7412
    assert rtps.user_unicast_port(0, 1) == 7413
    assert rtps.spdp_multicast_port(5) == 8650
    assert rtps.describe_port(7413) == "domain 0 participant 1 user unicast"
    assert rtps.describe_port(7410) == "domain 0 participant 0 discovery unicast"
    assert rtps.describe_port(7650) == "domain 1 discovery multicast"


def test_ros_names():
    assert rtps.ros_topic_to_dds("/chatter") == "rt/chatter"
    assert rtps.ros_service_to_dds("/add_two_ints") == ("rq/add_two_intsRequest", "rr/add_two_intsReply")
    assert rtps.dds_to_ros("rr/add_two_intsReply") == ("service reply", "/add_two_ints")
    assert rtps.dds_to_ros("DCPSParticipant") is None
    assert rtps.ros_type_to_dds("std_msgs/msg/String") == "std_msgs::msg::dds_::String_"
    assert rtps.dds_type_to_ros("example_interfaces::srv::dds_::AddTwoInts_Request_") == "example_interfaces/srv/AddTwoInts_Request"


@pytest.mark.parametrize("members", [[], [5], [5, 6, 9], list(range(10, 266, 3))])
def test_sequence_number_sets(members):
    raw = rtps._seqset_pack(5 if not members else members[0], members)
    base, decoded, end = rtps._seqset_unpack(raw, 0)
    assert decoded == members and end == len(raw)


def test_message_round_trip():
    prefix = bytes(range(12))
    qos = rtps.ParameterList().add(rtps.PID_KEY_HASH, bytes(16))
    msg = (
        rtps.MessageBuilder(prefix)
        .info_ts(1234.5)
        .info_dst(bytes(range(1, 13)))
        .data(0x104, 0x103, 7, b"\x00\x01\x00\x00abc\x00", qos)
        .heartbeat(0x104, 0x103, 3, 7, 2)
        .acknack(0x104, 0x103, 5, [5, 7], 4)
        .gap(0x104, 0x103, 2, 3)
    )
    parsed = rtps.parse(bytes(msg))
    assert parsed.prefix == prefix and parsed.vendor == rtps.VENDOR_UGS
    received = rtps.interpret(parsed)
    data, hb, ack, gap = (r.submessage for r in received)
    assert received[0].timestamp == pytest.approx(1234.5)
    assert received[0].dest_prefix == bytes(range(1, 13))
    assert (data.reader, data.writer, data.seq, data.payload) == (0x104, 0x103, 7, b"\x00\x01\x00\x00abc\x00")
    assert data.inline_qos.get(rtps.PID_KEY_HASH) == bytes(16)
    assert (hb.first, hb.last, hb.count, hb.final) == (3, 7, 2, False)
    assert (ack.base, ack.missing, ack.count, ack.final) == (5, [5, 7], 4, True)
    assert (gap.start, gap.list_base) == (2, 3)


def test_data_frag_round_trip():
    sample = bytes(range(256)) * 10
    prefix = bytes(12)
    parts = {}
    for first in (1, 2, 3):
        msg = rtps.parse(bytes(rtps.MessageBuilder(prefix).data_frag(0, 0x103, 1, first, 1024, sample)))
        frag = rtps.interpret(msg)[0].submessage
        assert frag.sample_size == len(sample) and frag.fragment_size == 1024
        parts[frag.first_fragment] = frag.payload
    assert b"".join(parts[i] for i in (1, 2, 3)) == sample


def test_discovery_data_round_trip():
    pd = rtps.ParticipantData(
        guid_prefix=bytes(range(12)),
        metatraffic_unicast=[rtps.Locator.udpv4("10.0.0.2", 7410)],
        default_unicast=[rtps.Locator.udpv4("10.0.0.2", 7411)],
        builtin_endpoints=0x3F,
        entity_name="week05",
        user_data=b"enclave=/;",
    )
    back = rtps.ParticipantData.decode(pd.encode())
    assert back.guid_prefix == pd.guid_prefix and back.enclave == "/" and back.entity_name == "week05"
    assert [str(x) for x in back.default_unicast] == ["udpv4:10.0.0.2:7411"]
    assert back.builtin_endpoint_names()[:2] == ["participant announcer", "participant detector"]
    ep = rtps.EndpointData(
        rtps.Guid(bytes(range(12)), 0x1203),
        "rt/topic",
        "std_msgs::msg::dds_::String_",
        rtps.EndpointQos(reliability="reliable", durability="transient_local", depth=7),
        writer=True,
        user_data=b"typehash=RIHS01_abc;",
    )
    back = rtps.EndpointData.decode(ep.encode(), writer=True)
    assert (back.topic, back.type_name, back.type_hash) == ("rt/topic", "std_msgs::msg::dds_::String_", "RIHS01_abc")
    assert (back.qos.reliability, back.qos.durability, back.qos.depth) == ("reliable", "transient_local", 7)


def test_pcap_round_trip(tmp_path):
    grams = [pcap.Datagram(1.5, "127.0.0.1", 4000, "239.255.0.1", 7400, b"RTPS" + bytes(16))]
    pcap.write(tmp_path / "x.pcap", grams)
    assert list(pcap.read(tmp_path / "x.pcap")) == grams


def test_not_rtps():
    assert rtps.parse(b"hello") is None
    assert dissect(b"hello").summary == ["not RTPS"]


@pytest.mark.skipif(not (CAPTURES / "pubsub.pcap").is_file(), reason="no capture")
def test_tutorial_capture_decodes():
    """The talker and listener's packets: discovery, endpoints, then Hello World."""
    participants, endpoints, hello = {}, {}, []
    types = {}
    for dg in pcap.read(CAPTURES / "pubsub.pcap"):
        msg = rtps.parse(dg.payload)
        assert msg is not None and msg.vendor_name == "eProsima Fast DDS"
        for rec in rtps.interpret(msg):
            s = rec.submessage
            if not isinstance(s, rtps.Data) or s.payload is None:
                continue
            if s.writer == rtps.ENTITYID_SPDP_WRITER:
                pd = rtps.ParticipantData.decode(s.payload)
                participants[pd.guid_prefix] = pd
            elif s.writer in (rtps.ENTITYID_SEDP_PUBLICATIONS_WRITER, rtps.ENTITYID_SEDP_SUBSCRIPTIONS_WRITER):
                ep = rtps.EndpointData.decode(s.payload, s.writer == rtps.ENTITYID_SEDP_PUBLICATIONS_WRITER)
                endpoints[ep.guid] = ep
                types[rtps.Guid(rec.source_prefix, ep.guid.entity)] = ep.type_name
            elif types.get(rtps.Guid(rec.source_prefix, s.writer)) == "std_msgs::msg::dds_::String_":
                hello.append(cdr.deserialize("std_msgs/msg/String", s.payload)["data"])
    assert len(participants) >= 2
    assert all(p.enclave == "/" for p in participants.values())
    topic = [e for e in endpoints.values() if e.topic == "rt/topic"]
    assert {e.writer for e in topic} == {True, False}
    assert all(e.type_hash == "RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18" for e in topic)
    # the rclpy default profile; history depth is not part of what SEDP announces
    assert all((e.qos.reliability, e.qos.durability) == ("reliable", "volatile") for e in topic)
    numbers = [int(h.rsplit(" ", 1)[1]) for h in hello if h.startswith("Hello World: ")]
    assert len(numbers) >= 3
    assert sorted(set(numbers)) == list(range(min(numbers), max(numbers) + 1))


@pytest.mark.skipif(not (CAPTURES / "srvcli.pcap").is_file(), reason="no capture")
def test_service_capture_correlates_request_and_reply():
    requests, replies = [], []
    for dg in pcap.read(CAPTURES / "srvcli.pcap"):
        for rec in rtps.interpret(rtps.parse(dg.payload)):
            s = rec.submessage
            if isinstance(s, rtps.Data) and s.inline_qos is not None:
                ident = s.inline_qos.get(rtps.PID_RELATED_SAMPLE_IDENTITY)
                if ident is None or s.payload is None:
                    continue
                if len(s.payload) >= 20:
                    requests.append((ident, s, cdr.deserialize("example_interfaces/srv/AddTwoInts_Request", s.payload)))
                else:
                    replies.append((ident, s, cdr.deserialize("example_interfaces/srv/AddTwoInts_Response", s.payload)))
    assert requests and replies
    (req_id, req, req_value), (rep_id, _, rep_value) = requests[0], replies[0]
    assert req_value == {"a": 2, "b": 3} and rep_value == {"sum": 5}
    # the request names the client's reply reader; the reply quotes it with the request's sequence number
    assert rep_id[:16] == req_id[:16]
    assert rtps.seq_unpack(rep_id, 16) == req.seq


@pytest.mark.skipif(not (CAPTURES / "pubsub.pcap").is_file(), reason="no capture")
def test_dissector_covers_every_byte_of_the_header():
    lines = list(dissect_pcap(CAPTURES / "pubsub.pcap", verbose=True, limit=40))
    assert any("SPDP: participant" in line for line in lines)
    assert any("[   0:   4] magic: RTPS" in line for line in lines)
