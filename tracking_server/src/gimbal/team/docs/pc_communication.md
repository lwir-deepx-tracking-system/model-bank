# Runtime–PC communication

Runtime follows `pc-application`'s current `feat/gui-video` wire contract and the
adopted decisions in [Application 설계](https://app.notion.com/p/3f29de3f14a280829f90dc2dfd666d04).
The [1103 roadmap](https://app.notion.com/p/40c9de3f14a28074905bd2239d6e9fdb)
describes wider state/control/gimbal work. The [tracking handler](auto_control.md)
connects AUTO(ID) and keyboard MOVE to one Waveshare Serial executor, independently
of tracking frame arrival for manual motion and watchdog checks. The default
`deferred` handler remains communication-only; HOME completion is unimplemented.
The wire contract has no version field or YAML protocol selector.
The PC repository is not modified by this change.

## Configuration and responsibilities

All YAML parsing/validation remains in AppConfig (`config_parser`). Components
receive typed configuration, never YAML nodes or filenames.

```yaml
gui:
  enabled: true
  video:
    host: <PC IPv4>
    port: 5000
    codec: h264
    encoder: mpph264enc
    # Keep bitrate_kbps, jpeg_quality, keyframe_interval, fps and rtp_mtu.
  metadata:
    enabled: true
    host: <PC IPv4>
    port: 5001
    max_packet_bytes: 1200
  command:
    enabled: true
    handler: deferred
    bind_address: 0.0.0.0
    port: 5002
    receive_timeout_ms: 1000
```

This is an excerpt; retain the other required keys in `config/runtime.yaml`.
On the PC, set `control.host` to the board IP and select the same video codec.
Set GUI enablement and destination/listener addresses in `config/runtime.yaml`
for the deployment.
JPEG remains selectable with `codec: jpeg` / `encoder: mppjpegenc`; Docker CPU
transport tests explicitly select `x264enc` or `jpegenc`. No automatic encoder
fallback is performed.

`IGuiProtocol` separates wire encoding from transport. The single supported
implementation, `PcGuiProtocol`, is constructed directly; no protocol enum,
factory or YAML selector is needed. `GuiCommandHandlerFactory` creates the
`IGuiCommandHandler` application strategy selected by `command.handler`.
The `tracking` handler adds AUTO(ID) following through a shared executor; see [AUTO control](auto_control.md).
Unknown handler settings fail during configuration; an omitted handler defaults
to `deferred`; `tracking` requires control to be enabled and handles both AUTO
and manual commands. The removed `manual` setting is rejected during configuration.

`GuiSender` owns video/metadata transport. `GuiReceiver` owns the TCP session,
validates framing before allocating/reading the body, and delegates replies to
the handler. `GuiReceiverThread` owns the worker lifecycle. Only the worker uses
the command handler; stop() interrupts network waits from another thread.

## Metadata UDP 5001

All multibyte fields are big endian; floats use IEEE-754 float32 bits.

| Offset | Field | Bytes |
|---|---|---|
| 0 | Magic `LWIR` (0x4c574952) | 4 |
| 4 | Capture frame ID, nonzero uint64 | 8 |
| 12 | RTP timestamp, uint32 | 4 |
| 16 | GUI send start, epoch microseconds, uint64 | 8 |
| 24 | Track count, uint32 | 4 |
| 28 | Repeated track records | 25 each |

Each track is `ID:uint32, x:float32, y:float32, width:float32,
height:float32, class:uint8, confidence:float32`, in that order.
Wire IDs retain the full uint32 range (1..4,294,967,295). Tracker int IDs are
adapted only at the sender boundary; invalid signed IDs are excluded, never cast
into large valid IDs. Internal class -1 maps to PC Unknown (255); class values
outside [-1,255] cannot be represented and that record is excluded. Class IDs
80..254 are preserved rather than changed because the PC lacks a name mapping.
Invalid geometry/confidence records are excluded individually; valid boxes are
clipped to original 640×480 bounds. The internal tracker output is unchanged.

The PC assumes **original 640×480 pixels**. Video and metadata have no resize or
coordinate conversion on the sender; incompatible source dimensions are
rejected. Width/height, version, message type and detection count are not sent.
The original capture ID is preserved, while the RTP timestamp is computed from
the video submission index and configured FPS at 90 kHz. Gaps in capture IDs do
not change that mapping. `gui_started_us` alone does not measure network latency
without a clock synchronization contract.

A datagram carries display records for one frame, including a header for zero
records. At the default 1200-byte limit, 46 tracks fit (1178 bytes); 47 would need
1203 bytes. Capacity is derived from the configured payload limit, not a fixed
track count. Valid records are included in stable input order up to capacity;
extra display records are excluded, and trackCount is the actual encoded count.
The local metadata object supports a selected-target priority hint, which is not
transmitted. When a valid selected target is supplied it is included first, then
the other records in stable order. Current deferred control supplies no target
hint because input-ID following is unimplemented. Display exclusion does not
mean TargetLost or affect tracking/control state. No fragmentation is added.

The adopted overflow rule is in Application 설계 §2.5; the roadmap's §§5/7 still
list that policy as unresolved. This change follows the explicitly adopted design
for selection; the numeric UDP limit remains a provisional, unmeasured setting.
Actual UDP send errors still propagate as sender failure; transient transport
error recovery is not added here.

## Control TCP 5002

Header: `LWIR:uint32, category:uint8, command:uint8, payload_length:uint16`.
There is no version field. Reads may split or combine messages arbitrarily.
Unsupported category/command/length or a board-only message from the PC closes
the client; the listener remains available for reconnection.

| Category | Command | Payload | Deferred strategy behavior |
|---|---|---|---|
| System (1) | PING (4) | empty | ACK OK |
| Mode (2) | AUTO (1) | Track ID uint32 | ACK NOT_READY; zero ID gets INVALID_TARGET |
| Mode (2) | MANUAL (2) | empty | ACK OK, already Manual with no target |
| Gimbal (3) | MOVE (1) | signed x,y bytes (-1..1) | No operation; no ACK per PC contract |
| Gimbal (3) | HOME (3) | empty | ACK NOT_READY |

On every connection the deferred handler sends STATUS (System command 3):
`Manual / None / Idle / error 0 / selected ID 0`. No tracking or motion is active.
The state payload is `mode:uint8, target_state:uint8, gimbal_state:uint8,
error:uint8, selected_ID:uint32` (8 bytes). ACK (System command 1) contains the
original category, original command and result (3 bytes).

The codec also supports INIT (System command 2), with the same state payload.
The deferred handler has no reinitialization event or state transition to report;
it does not invent periodic STATUS, AUTO success or HOME completion.

**The default deferred strategy does not follow targets or complete HOME.**
Select the [tracking handler](auto_control.md) for AUTO(ID) following and manual override.
It preserves the wire format and uses a single device executor; HOME remains NOT_READY.
The behavior in the deferred table applies only to `handler: deferred`. Received
AUTO IDs retain all 32 bits in both strategies. The tracking handler validates IDs
against fresh tracker output and owns real transitions through the shared executor;
see [AUTO control](auto_control.md) for readiness, STATUS, disconnect, and heartbeat
rules. The deferred handler has no moving state and therefore implements no motion
watchdog.

## Pipeline extension — TCP 5002

This runtime adds category **Pipeline (4)** to the existing 8-byte header.
PC software must implement this extension; older PC applications remain usable
with the existing commands. All integers below are unsigned and big endian.
Only counts and switch results are sent; no names, paths, model lists, or tracker
configuration records are transmitted. AppConfig requires both catalogs to use
consecutive IDs `1..count`. Request IDs retain all 32 bits, including zero.

| Direction | Command | Payload |
|---|---|---|
| PC → board | Model count `0x01` | request_id:u32 (4 bytes) |
| PC → board | Tracker count `0x02` | request_id:u32 (4 bytes) |
| PC → board | Change model `0x03` | request_id:u32, model_id:u32 (8 bytes) |
| PC → board | Change tracker `0x04` | request_id:u32, tracker_id:u32 (8 bytes) |
| Board → PC | Model count `0x81` | request_id:u32, count:u32 (8 bytes) |
| Board → PC | Tracker count `0x82` | request_id:u32, count:u32 (8 bytes) |
| Board → PC | Model result `0x83` | request_id:u32, phase:u8, result:u8 (6 bytes) |
| Board → PC | Tracker result `0x84` | request_id:u32, phase:u8, result:u8 (6 bytes) |

Phase is `1=Accepted`, `2=Completed`. Results are `0=OK`, `1=InvalidId`,
`2=Busy`, `3=CreationFailed`, `4=Restored` (previous model),
`5=RestoreFailed`, `6=NotReady` (startup/shutdown or unavailable request service).
These results are separate from the existing System ACK result enum. Pipeline
commands do not also emit System ACK. Rejected requests receive Completed only;
accepted requests receive Accepted/OK followed by their actual Completed result.
An already-active ID succeeds without recreating the instance and follows the
same two-response contract. A tracker creation failure preserves its previous
instance. A failed model load reports Restored only after rollback succeeds.

For example, request 10 changes to model 2:

```text
PC:       4c574952 04 03 0008 0000000a 00000002
Accepted: 4c574952 04 83 0006 0000000a 01 00
Complete: 4c574952 04 83 0006 0000000a 02 00
```

`PipelineCommandHandler` wraps the configured deferred/tracking handler and
associates with `IPipelineRequests`, implemented by the cpp-private runtime.
The TCP receiver worker validates and dispatches requests, sends acceptance,
and polls completion futures at most every 25 ms while continuing PING and
partial-packet reception. The model manager performs model teardown/load/rollback;
the runtime loop performs tracker create/swap. These workers never write the TCP
socket; the receiver worker sends every control reply. Video/metadata sender
workers keep their existing responsibilities and the wire metadata is unchanged.

At most one model change and one tracker change may be outstanding globally.
The handler additionally limits a session to two pending changes and rejects a
duplicate pending change request_id as Busy. Clients should not reuse outstanding
IDs. Disconnect drops that session's response subscriptions but does not cancel
an accepted operation. Its result is never delivered to a replacement session;
a reconnecting client may query counts again and receives Busy until the actual
outstanding operation completes. Count queries and PING do not wait for switching.
Without an associated runtime request service, count queries return zero and
switches return NotReady. Board-only Pipeline responses sent by a PC, unsupported
commands, and incorrect declared payload lengths disconnect that client as before.

## Verification scope

Tests use PC source fixtures for exact metadata and control bytes, TCP loopback
for split/coalesced messages, reconnect and stop, and CPU software encoders for
RTP timestamp matching. Test frames and track records are explicit transport
fixtures, not inference results. These checks do not establish Windows GUI
end-to-end behavior or board hardware performance.

DXNN loading, inference, tensor validation, NPU benchmarks and board MPP/gimbal
execution require actual hardware verification.
