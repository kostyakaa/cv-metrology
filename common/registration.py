"""Whole-scan translation adjustment using GRBL positions, optical displacement and surveyed marks."""
import cv2
import numpy as np

from common.markers import detect_precise, make_detector
from common.odometry import TileFeatures, plane_points


def neighbours(positions):
    pairs = {(i, i+1) for i in range(len(positions)-1)}
    for i, p in enumerate(positions):
        distance = np.linalg.norm(positions-p, axis=1)
        for j in np.argsort(distance, kind='stable')[:9]:
            if i != j:
                pairs.add(tuple(sorted((i, int(j)))))
    return sorted(pairs)


def adjust(positions, edges, anchors, grbl_prior=True):
    """Robust XY graph fit. Without pose priors the caller must anchor every component."""
    n = len(positions)
    i = np.array([e['i'] for e in edges], int)
    j = np.array([e['j'] for e in edges], int)
    raw = np.array([e['delta_mm'] for e in edges], float).reshape(-1, 2) - (positions[j]-positions[i])
    sigma = np.array([max(.13, e['scatter_mm']*2) for e in edges])
    nodes = np.array([a['i'] for a in anchors], int)
    target = np.array([a['offset'] for a in anchors]).reshape(-1, 2)
    offset = np.zeros_like(positions, dtype=float)
    for _ in range(8):
        residual = np.linalg.norm(offset[j]-offset[i]-raw, axis=1)
        w = np.minimum(1., .20/np.maximum(residual, 1e-8))/sigma**2
        wa = np.minimum(1., .20/np.maximum(np.linalg.norm(offset[nodes]-target, axis=1), 1e-8))/.10**2
        A = np.eye(n)/.45**2 if grbl_prior else np.zeros((n,n))
        b = np.zeros_like(offset)
        np.add.at(A, (i, i), w)
        np.add.at(A, (j, j), w)
        np.add.at(A, (i, j), -w)
        np.add.at(A, (j, i), -w)
        np.add.at(b, i, -raw*w[:, None])
        np.add.at(b, j, raw*w[:, None])
        np.add.at(A, (nodes, nodes), wa)
        np.add.at(b, nodes, target*wa[:, None])
        updated = np.linalg.solve(A, b)
        if np.max(np.abs(offset-updated)) < .0001:
            return updated
        offset = updated
    return offset


def optical_coordinates(tags, measured, records):
    """Rejoin interrupted optical tracks across rows without machine displacements or marker layout."""
    graph = []
    adjacent = [set() for _ in tags]
    for e in measured:
        i,j = e['i'],e['j']
        ids = tags[i].keys() & tags[j].keys()
        if ids:
            step = np.median([tags[i][k]-tags[j][k] for k in ids],axis=0)
        elif e.get('tile'):
            step = e['tile']['delta_mm']
        elif j == i+1 and records[j].get('motion_from_previous',{}).get('valid'):
            step = records[j]['motion_from_previous']['delta_mm']
        else:
            continue
        graph.append(dict(i=i,j=j,delta_mm=step,scatter_mm=.10))
        adjacent[i].add(j)
        adjacent[j].add(i)
    roots, anchors = {}, []
    for i in range(len(tags)):
        if i in roots:
            continue
        pending = [i]
        roots[i] = i
        anchors.append(dict(i=i,offset=[0,0]))
        while pending:
            for j in adjacent[pending.pop()]:
                if j not in roots:
                    roots[j] = i
                    pending.append(j)
    xy = adjust(np.zeros((len(tags),2)),graph,anchors,grbl_prior=False)
    return [dict(origin_frame=roots[i],xy_mm=(p-xy[roots[i]]).tolist()) for i,p in enumerate(xy)]


def residual_stats(vectors):
    if not len(vectors):
        return None
    lengths = np.linalg.norm(vectors, axis=1)
    return dict(count=len(lengths), rms_mm=float(np.sqrt(np.mean(lengths**2))),
                p95_mm=float(np.percentile(lengths, 95)))


def projected_points(position, local, scale):
    xy = position+local
    if scale is not None:
        # Invert the same dest->source mapping used by the renderer.
        for _ in range(5):
            xy = position+local*(1+scale.at(xy)[...,None])
    return xy


def navigation_quality(meta):
    """Coarse binding gate. Passing this check does not certify positioning accuracy."""
    reasons = []
    if not meta.get('complete'):
        reasons.append('маршрут не завершён')
    closure = np.asarray(meta.get('closure_command_mm'), dtype=float)
    if closure.shape != (2,) or not np.isfinite(closure).all() or np.linalg.norm(closure) > .5:
        reasons.append('замыкание отсутствует или превышает 0,5 мм')
    if 'return_to_start_mm' in meta:
        drift = np.asarray(meta['return_to_start_mm'], float)
        if drift.shape != (2,) or not np.isfinite(drift).all() or np.linalg.norm(drift) > .5:
            reasons.append('возврат к начальному наблюдению превышает 0,5 мм')
    reg = meta.get('registration') or {}
    ids = sorted({a['marker'] for a in reg.get('anchors', [])})
    if len(ids) < 3:
        reasons.append('меньше трёх различных опорных меток')
    return dict(usable=not reasons, reasons=reasons, marker_ids=ids)


def marker_anchors(observations, meta):
    """Older scans save corner coordinates without IDs. Resolve only unique near-vertex matches."""
    polygon = np.array(meta['polygon'])
    layout = meta.get('marker_positions', {})
    groups = {}
    for i, mid, xy in observations:
        groups.setdefault(mid, []).append((i, xy))
    targets = {}
    for mid, sightings in groups.items():
        if str(mid) in layout:
            targets[mid] = np.array(layout[str(mid)])
        elif not layout and len(sightings) >= 2:
            near = np.linalg.norm(polygon-np.median([xy for _, xy in sightings], axis=0), axis=1) < 1.5
            if near.sum() == 1:
                targets[mid] = polygon[near][0]
    anchors = []
    for mid, ref in targets.items():
        if any(np.linalg.norm(ref-p) < 1.5 for k, p in targets.items() if k != mid):
            continue
        for i, xy in groups[mid]:
            if np.linalg.norm(ref-xy) <= 1.5:
                anchors.append(dict(i=i, marker=mid, offset=(ref-xy).tolist()))
    return anchors


class ProjectionScale:
    """Local change of image-to-plane scale supported by travel in two independent directions."""
    spacing = 4.

    def __init__(self, polygon, positions, edges, anchors):
        self.origin = np.min(polygon, axis=0)-self.spacing
        w, h = (np.ceil((np.max(polygon, axis=0)-self.origin)/self.spacing).astype(int)+2)
        weight = np.zeros((h, w), np.float64)
        total = np.zeros_like(weight)
        moments = np.zeros((h, w, 3))
        for e in edges:
            tile = e['tile']
            if tile is None:
                continue
            d = np.array(tile['delta_mm'])
            length = np.linalg.norm(d)
            if length < 5:
                continue
            unit = d/length
            g = e['grbl_delta_mm']
            scale = np.dot(g, unit)/length-1
            if abs(scale) > .04 or np.linalg.norm(g-(1+scale)*d) > .4:
                continue
            pts = positions[e['i']]+tile['local_support_mm']
            grid = np.unique(np.rint((pts-self.origin)/self.spacing).astype(int), axis=0)
            x, y = grid.T
            inside = (x>=0)&(x<w)&(y>=0)&(y<h)
            x, y = x[inside], y[inside]
            weight[y,x] += 1
            total[y,x] += scale
            moments[y,x] += [unit[0]**2, unit[0]*unit[1], unit[1]**2]
        def smooth(a):
            return cv2.GaussianBlur(a, (0, 0), 1.5, borderType=cv2.BORDER_CONSTANT)
        a,b,c = smooth(moments).transpose(2,0,1)
        # Smaller eigenvalue counts support perpendicular to the dominant travel.
        support = .5*(a+c-np.sqrt((a-c)**2+4*b*b))
        field = smooth(total)/np.maximum(smooth(weight), 1.e-6)*np.clip(support, 0, 1)
        yy, xx = np.indices(weight.shape)
        for anchor in anchors:
            x, y = (np.array(anchor['target'])-self.origin)/self.spacing
            # A surveyed 15 mm marker is on the reference plane.
            radius = np.hypot(xx-x, yy-y)*self.spacing
            field[radius < anchor['radius_mm']] = 0
        self.field = field.astype(np.float32)

    def at(self, points):
        xy = (np.asarray(points)-self.origin)/self.spacing
        x,y = xy[...,0],xy[...,1]
        ix,iy = np.floor(x).astype(int),np.floor(y).astype(int)
        valid = (ix>=0)&(ix<self.field.shape[1]-1)&(iy>=0)&(iy<self.field.shape[0]-1)
        fx,fy = x-ix,y-iy
        ix=np.clip(ix,0,self.field.shape[1]-2)
        iy=np.clip(iy,0,self.field.shape[0]-2)
        f=self.field
        return valid*((1-fy)*((1-fx)*f[iy,ix]+fx*f[iy,ix+1])+fy*((1-fx)*f[iy+1,ix]+fx*f[iy+1,ix+1]))

    def project_edges(self, edges, positions):
        projected = []
        for e in edges:
            scale = 0.
            if e['tile']:
                xy = positions[e['i']]+e['tile']['local_support_mm']
                scale = float(np.median(self.at(xy)))
            projected.append(dict(e, delta_mm=(np.array(e['delta_mm'])*(1+scale)).tolist()))
        return projected


def register(folder, meta, K, D, model):
    records = meta['records']
    positions = np.array([r['position'] for r in records], float)
    if positions.shape != (len(records), 2) or not len(records) or not np.isfinite(positions).all():
        raise ValueError('Expected a non-empty scan with finite GRBL XY positions')
    features, observations = [], []
    fast, precise = make_detector(), make_detector(precise=True)
    sift = cv2.SIFT_create(nfeatures=1600, contrastThreshold=.012)
    for i, r in enumerate(records):
        path = folder/r['file']
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f'Cannot read {path}')
        features.append(TileFeatures(gray, K, D, model, r, sift))
        for mid, corners in detect_precise(fast, precise, gray).items():
            if mid in range(4):
                xy = plane_points(corners, K, D, model, np.array(r['H'])).mean(axis=0)
                observations.append((i, mid, xy))
    anchors = marker_anchors(observations, meta)
    accepted = {(a['i'], a['marker']) for a in anchors}
    layout = meta.get('marker_positions', {})
    marker_observations = []
    for i, mid, xy in observations:
        target = layout.get(str(mid))
        residual = None if target is None else (xy - np.asarray(target)).tolist()
        marker_observations.append(dict(frame=i, marker=mid, xy=xy.tolist(), residual_mm=residual,
                                        accepted=(i, mid) in accepted))
    for anchor in anchors:
        i, mid = anchor['i'], anchor['marker']
        xy = next(p for k, m, p in observations if k == i and m == mid)
        anchor['target'] = (xy+anchor['offset']).tolist()
        anchor['radius_mm'] = float(meta.get('marker_size_mm', 15))/np.sqrt(2)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    edges, measured = [], []
    for i, j in neighbours(positions):
        tile = features[i].displacement(features[j], matcher)
        flow = records[j].get('motion_from_previous') if j == i+1 else None
        e = dict(i=i, j=j, grbl_delta_mm=(positions[j]-positions[i]).tolist(), tile=tile)
        valid_flow = flow and flow.get('valid') and flow.get('timing_consistent',True)
        choice = tile
        if valid_flow and tile and np.linalg.norm(np.subtract(tile['delta_mm'], flow['delta_mm'])) > .5:
            choice = None
            e['reason'] = 'endpoint and motion disagree'
        elif valid_flow and not tile:
            choice = dict(delta_mm=flow['delta_mm'], scatter_mm=max(.20, .05*np.sqrt(len(flow['steps']))))
        if choice:
            e.update(choice)
            e['source'] = 'endpoints' if tile else 'motion'
            if tile and valid_flow:
                # Same camera, correlated observations. Give the stop-to-stop match more weight.
                e['delta_mm'] = (.8*np.array(tile['delta_mm'])+.2*np.array(flow['delta_mm'])).tolist()
                e['source'] = 'endpoints+motion'
            discrepancy = np.subtract(e['delta_mm'], e['grbl_delta_mm'])
            e['visual_minus_grbl_mm'] = discrepancy.tolist()
            length = np.linalg.norm(e['grbl_delta_mm'])
            if length > 0:
                e['relative_disagreement'] = float(np.linalg.norm(discrepancy)/length)
            if np.linalg.norm(discrepancy) <= min(2., .5+.05*length):
                edges.append(e)
            else:
                e['reason'] = 'outside fusion gate'
        e['accepted'] = choice is not None and 'reason' not in e
        measured.append(e)

    # Withhold complete pairs and marker observations for a repeatable cross-check before the final fit.
    test = edges[4::5]
    train = [e for k, e in enumerate(edges) if k % 5 != 4]
    tag_test = anchors[2::3]
    train_tags = [a for k, a in enumerate(anchors) if k % 3 != 2]
    cv_scale = ProjectionScale(meta['polygon'], positions, train, train_tags)
    cv_offset = adjust(positions, cv_scale.project_edges(train, positions), train_tags)
    def edge_errors(offset, scale=None):
        p = positions+offset
        differences = []
        for e in test:
            i, j, tile = e['i'], e['j'], e['tile']
            if tile:
                a = projected_points(p[i], np.asarray(tile['local_support_mm']), scale)
                b = projected_points(p[j], np.asarray(tile['other_support_mm']), scale)
                differences.append(np.median(b-a, axis=0))
            else:
                differences.append(p[j]-p[i]-e['delta_mm'])
        return residual_stats(differences)
    def tag_errors(offset, scale=None):
        differences = []
        for a in tag_test:
            i, target = a['i'], np.array(a['target'])
            local = target-a['offset']-positions[i]
            differences.append(projected_points(positions[i]+offset[i], local, scale)-target)
        return residual_stats(differences)
    zero = np.zeros_like(positions)
    checks = dict(edges_grbl=edge_errors(zero), edges_fused=edge_errors(cv_offset,cv_scale),
                  markers_grbl=tag_errors(zero), markers_fused=tag_errors(cv_offset,cv_scale))
    scale = ProjectionScale(meta['polygon'], positions, edges, anchors)
    offsets = adjust(positions, scale.project_edges(edges, positions), anchors)
    # Keep a camera-only trajectory in segment-local coordinates. Never fill lost motion with GRBL.
    consecutive = {e['j']: e for e in measured if e['j'] == e['i']+1}
    trajectory, segment, xy = [], 0, np.zeros(2)
    tags = [{} for _ in records]
    for i, mid, world in observations:
        tags[i][mid] = world-positions[i]
    for j in range(len(records)):
        source = None
        if j:
            e = consecutive[j]
            if 'delta_mm' in e:
                xy += e['delta_mm']
                source = e['source']
            elif tags[j-1].keys() & tags[j].keys():
                ids = tags[j-1].keys() & tags[j].keys()
                xy += np.median([tags[j-1][m]-tags[j][m] for m in ids],axis=0)
                source = 'common marker'
            elif records[j].get('motion_from_previous', {}).get('valid'):
                xy += records[j]['motion_from_previous']['delta_mm']
                source = 'motion'
            else:
                segment += 1
                xy = np.zeros(2)
        trajectory.append(dict(segment=segment, xy_mm=xy.tolist(), source=source))
    return offsets, scale, dict(method='optical-grbl-plane-v1', pairs=measured, anchors=anchors,
                         marker_observations=marker_observations,
                         scale_grid=dict(origin=scale.origin.tolist(), spacing=scale.spacing, values=scale.field.tolist()),
                         optical_trajectory=trajectory, optical_positions=optical_coordinates(tags,measured,records),
                         validation=checks)

