"""Supplied-track measurement, not automatic tracking or mask morphology."""
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import re
import xml.etree.ElementTree as ET

VERSION = '0.1.0-dev.1'

@dataclass(frozen=True)
class Calibration:
    um_per_pixel: float | None = None
    seconds_per_frame: float | None = None
    spatial_source: str = ''
    temporal_source: str = ''

    def validate(self):
        for v, source in ((self.um_per_pixel, self.spatial_source), (self.seconds_per_frame, self.temporal_source)):
            if v is not None and (isinstance(v, bool) or not math.isfinite(v) or v <= 0 or not source.strip()):
                raise ValueError('Positive finite calibration and its documentary source are required.')

def import_cimrak(payload, *, sequence_id):
    if not sequence_id.strip():
        raise ValueError('Sequence/experiment ID is required. Do not pool independent recordings.')
    if not payload or len(payload) > 8 * 1024 * 1024:
        raise ValueError('XML must be nonempty and at most 8 MiB.')
    if b'<!DOCTYPE' in payload.upper() or b'<!ENTITY' in payload.upper():
        raise ValueError('DTD/entity declarations are unsupported.')
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError('Malformed XML.') from exc
    if root.tag != 'data' or root.find('images') is None:
        raise ValueError('Require the documented Cimrak data/images schema.')
    images = root.findall('images/image')
    if not images or len(images) > 10000:
        raise ValueError('Require 1–10000 annotated source frames.')
    rows, frames, seen = [], [], set()
    for image in images:
        src = image.findtext('src', '')
        match = re.fullmatch(r'image(\d+)\.png', src.replace('\\', '/').split('/')[-1], flags=re.I)
        if not match:
            raise ValueError('Unsupported frame filename; no timing inferred from XML order.')
        frame = int(match.group(1))
        if frame < 1 or (frames and frame <= frames[-1]):
            raise ValueError('Source frame IDs must be unique and increasing; no automatic sorting.')
        frames.append(frame)
        boxes = image.find('boundingboxes')
        if boxes is None:
            raise ValueError('Missing boundingboxes element; not treated as an empty frame.')
        for box in boxes.findall('boundingbox'):
            try:
                x, y, w, h = (float(box.findtext(k, '')) for k in ('x_left_top', 'y_left_top', 'width', 'height'))
            except ValueError as exc:
                raise ValueError('Non-numeric bounding-box geometry.') from exc
            if not all(math.isfinite(v) and abs(v) <= 1e7 for v in (x, y, w, h)) or x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError('Invalid or out-of-budget bounding-box geometry.')
            track = box.findtext('class_name/track_id')
            track = track.strip() if track is not None else ''
            # Zero is a publisher track ID, NOT background or an untracked sentinel.
            if track and (frame, track) in seen:
                raise ValueError('Duplicate track ID within one frame; identity is ambiguous.')
            if track:
                seen.add((frame, track))
            rows.append(dict(sequence_id=sequence_id, frame=frame, source_frame=src,
                             track_id=track or None, x_left_top=x, y_left_top=y, width=w, height=h,
                             center_x_px=x+w/2, center_y_px=y+h/2,
                             annotation_status='PUBLISHER_TRACK' if track else 'UNTRACKED_ANNOTATION'))
            if len(rows) > 100000:
                raise ValueError('More than 100000 boxes; none silently omitted.')
    if not rows:
        raise ValueError('No boxes available.')
    return dict(rows=rows, frames=frames, sequence_id=sequence_id, source_sha256=sha256(payload).hexdigest(),
                coordinate_convention='Publisher top-left plus width/height; box center = top-left + half extent. Pixel-coordinate proxy, not mask centroid.',
                frame_convention='Numeric imageN.png source index retained; not acquisition seconds.',
                identity_convention='Literal publisher IDs including 0; missing ID is untracked, not repaired.')

def measure_tracks(imported, calibration=Calibration()):
    calibration.validate()
    groups = defaultdict(list)
    for row in imported['rows']:
        if row['track_id'] is not None:
            groups[row['track_id']].append(row)
    physical = calibration.um_per_pixel is not None and calibration.seconds_per_frame is not None
    tracks, steps = [], []
    for track, points in groups.items():
        path, gaps, adjacent = 0., 0, []
        for a, b in zip(points, points[1:]):
            delta_frame = b['frame']-a['frame']
            dx, dy = b['center_x_px']-a['center_x_px'], b['center_y_px']-a['center_y_px']
            displacement = math.hypot(dx, dy)
            contiguous = delta_frame == 1
            if contiguous:
                path += displacement
                adjacent.append(displacement)
            else:
                gaps += 1
            steps.append(dict(sequence_id=imported['sequence_id'], track_id=track,
                              start_frame=a['frame'], end_frame=b['frame'], delta_frame=delta_frame,
                              dx_px=dx, dy_px=dy, endpoint_displacement_px=displacement,
                              gap=not contiguous, path_included=contiguous,
                              displacement_px_per_frame=displacement if contiguous else None,
                              speed_um_per_s=displacement*calibration.um_per_pixel/calibration.seconds_per_frame if physical and contiguous else None,
                              motion_direction_rad=math.atan2(dy, dx) if contiguous and displacement > 0 else None,
                              note='Observed box-center step; not cell orientation' if contiguous else 'Gap: endpoint displacement only; not interpolated or included in observed path'))
        first, last = points[0], points[-1]
        span = last['frame']-first['frame']
        net = math.hypot(last['center_x_px']-first['center_x_px'], last['center_y_px']-first['center_y_px']) if len(points)>1 else None
        tracks.append(dict(sequence_id=imported['sequence_id'], track_id=track, observations=len(points),
                           first_frame=first['frame'], last_frame=last['frame'], observed_span_frames=span,
                           observed_span_s=span*calibration.seconds_per_frame if calibration.seconds_per_frame else None,
                           adjacent_steps=len(adjacent), gap_links=gaps, net_endpoint_displacement_px=net,
                           observed_adjacent_path_px=path if len(points)>1 else None,
                           mean_adjacent_displacement_px_per_frame=sum(adjacent)/len(adjacent) if adjacent else None,
                           maximum_adjacent_displacement_px_per_frame=max(adjacent) if adjacent else None,
                           status='GAPPED_PARTIAL_PATH' if gaps else 'SINGLE_OBSERVATION' if len(points)==1 else 'CONTIGUOUS_PUBLISHER_TRACK'))
    counts = dict(annotated_frames=len(imported['frames']), boxes=len(imported['rows']), tracks=len(tracks),
                  untracked_boxes=sum(r['track_id'] is None for r in imported['rows']), adjacent_steps=sum(s['path_included'] for s in steps),
                  gap_links=sum(s['gap'] for s in steps))
    return dict(version=VERSION, sequence_id=imported['sequence_id'], source_sha256=imported['source_sha256'],
                observations=imported['rows'], tracks=tracks, steps=steps, counts=counts,
                calibration=asdict(calibration), physical_velocity_status='AVAILABLE_FROM_SUPPLIED_CALIBRATIONS' if physical else 'NOT_EVALUABLE: documented spatial and time calibration both required',
                scope='EXPERIMENTAL supplied publisher-track coordinate measurements. Not automatic tracker accuracy, cell-mask shape, full-field density, fluid velocity, membrane mechanics or biological validation.',
                exclusions='No boxes removed. Untracked boxes retained; gap links excluded from path/rates but retained as endpoint observations.',
                notes=[imported[k] for k in ('coordinate_convention','frame_convention','identity_convention')])
