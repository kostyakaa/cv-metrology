"""Render a metric plane mosaic with a single source sample per output pixel."""
import cv2
import numpy as np


def pixel_to_machine(K, V, n, d, position):
    # Ray/plane intersection, then displacement required to put that point on optical axis.
    B = np.linalg.inv(np.column_stack([V, [0., 0., 1.]]))[:2]
    H = np.vstack([d * B, n]) @ np.linalg.inv(K)
    H[:2] += np.asarray(position)[:, None] * H[2]
    return H


def overlap_error(a, b, mask):
    """Translation of content a->b in already metric-warped images, with forward/backward checks."""
    ga, gb = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b)]
    mask = cv2.erode(mask.astype(np.uint8) * 255, np.ones((25, 25), np.uint8))
    p = cv2.goodFeaturesToTrack(ga, 500, .02, 12, mask=mask)
    if p is None or len(p) < 20:
        return dict(valid=False, reason='insufficient texture')
    q, ok, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p, None)
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, q, None)
    good = (ok.ravel() != 0) & (ok2.ravel() != 0) & (np.linalg.norm(back-p, axis=2).ravel() < .5)
    delta = (q-p).reshape(-1, 2)[good]
    if len(delta) < 20:
        return dict(valid=False, reason='inconsistent matches')
    med = np.median(delta, axis=0)
    keep = np.linalg.norm(delta-med, axis=1) < 1.
    valid = bool(keep.sum() >= 20 and keep.mean() >= .7)
    return dict(valid=valid, shift_px=med.tolist(), tracks=int(keep.sum()),
                scatter_px=float(np.median(np.linalg.norm(delta-med, axis=1))),
                reason='translation consensus' if valid else 'parallax or ambiguous texture')


def target_displacement(reference, live, coverage, ppm):
    """Measure content shift in a metric square centred at the selected mosaic pixel."""
    images = [cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32) for im in (reference,live)]
    a,b = [im-cv2.GaussianBlur(im,(0,0),max(1., .35*ppm)) for im in images]
    mid = a.shape[0]//2
    radius, search, spacing = (max(2,round(ppm*f)) for f in (1.6,1.6,2.4))
    shifts = []
    centres = ((0,0),(spacing,0),(-spacing,0),(0,spacing),(0,-spacing))
    centre_match = None
    for dx,dy in centres:
        x,y = mid+dx,mid+dy
        tpl = a[y-radius:y+radius+1,x-radius:x+radius+1]
        if tpl.std() < 2 or not coverage[y-radius:y+radius+1,x-radius:x+radius+1].all():
            continue
        region = b[y-radius-search:y+radius+search+1,x-radius-search:x+radius+search+1]
        scores = cv2.matchTemplate(region,tpl,cv2.TM_CCOEFF_NORMED)
        _, peak, _, (u,v) = cv2.minMaxLoc(scores)
        if peak < .67 or not (0<u<2*search and 0<v<2*search):
            continue
        # Repeated patterns and straight grooves can produce a broad ridge or another peak.
        yy,xx = np.ogrid[:scores.shape[0],:scores.shape[1]]
        competing = (xx-u)**2+(yy-v)**2 >= max(2., .25*ppm)**2
        if peak-float(scores[competing].max()) < .055:
            continue
        row, col = scores[v,u-1:u+2], scores[v-1:v+2,u]
        sub = [.5*float(s[0]-s[2])/min(-1e-6,float(s[0]-2*s[1]+s[2])) for s in (row,col)]
        shift = (np.array([u,v])+np.clip(sub,-.5,.5)-search)/ppm
        shifts.append(shift)
        if dx == dy == 0:
            centre_match = shift
    if len(shifts) < 2 or centre_match is None:
        return None
    centre = np.median(shifts,axis=0)
    inliers = np.linalg.norm(shifts-centre,axis=1) < .16
    if np.linalg.norm(centre_match-centre) >= .16 or inliers.sum() < max(2, .66*len(shifts)):
        return None
    return np.median(np.asarray(shifts)[inliers],axis=0)


class Mosaic:
    def __init__(self, polygon, ppm=10, blend_lighting=False):
        self.polygon = np.asarray(polygon, np.float32)
        self.origin = self.polygon.min(axis=0)
        self.ppm = ppm
        self.size = tuple((np.ceil(np.ptp(self.polygon, axis=0)*ppm)+1).astype(int))
        if np.prod(self.size) > 25_000_000:
            raise ValueError('Mosaic exceeds 25 megapixels; reduce --ppm')
        self.T = np.array([[ppm,0,-ppm*self.origin[0]], [0,ppm,-ppm*self.origin[1]], [0,0,1.]])
        self.weight = np.zeros(self.size[::-1], np.float32)
        self.best = np.zeros((*self.size[::-1],3), np.uint8)
        self.blend_lighting = blend_lighting
        if blend_lighting:
            self.light_sum = np.zeros_like(self.best, dtype=np.float32)
            self.light_weight = np.zeros_like(self.weight)
            self.source_light = np.zeros_like(self.best)
        self.region = np.zeros(self.size[::-1], np.uint8)
        cv2.fillConvexPoly(self.region, np.round((self.polygon-self.origin)*ppm).astype(np.int32), 1)
        self.previous = None
        self.errors = []

    def add(self, image, valid, H, position, raw=None, K=None, D=None, model='pinhole', diagnose=True,
            owner=None, tile_id=None, scale_map=None):
        transform = self.T @ H
        mask = cv2.warpPerspective(valid.astype(np.uint8), transform, self.size,
                                   flags=cv2.INTER_NEAREST) * self.region
        if raw is not None:
            # Compose plane projection and lens distortion before sampling source once.
            x,y,width,height=cv2.boundingRect(mask)
            if width == 0 or height == 0:
                return
            if scale_map is not None:
                # The source footprint can grow by up to 4% about the optical axis.
                centre = (np.asarray(position)-self.origin)*self.ppm
                pad = int(np.ceil(.05*max(abs(x-centre[0]), abs(x+width-centre[0]),
                                          abs(y-centre[1]), abs(y+height-centre[1]))))+2
                right, bottom = min(x+width+pad,self.size[0]), min(y+height+pad,self.size[1])
                x, y = max(0,x-pad), max(0,y-pad)
                width, height = right-x, bottom-y
            yy,xx=np.indices((height,width),dtype=np.float32)
            xx += x
            yy += y
            roi = np.s_[y:y+height, x:x+width]
            if scale_map is not None:
                scale = 1+scale_map[roi]
                xx = centre[0]+(xx-centre[0])/scale
                yy = centre[1]+(yy-centre[1])/scale
            points=np.stack([xx,yy],axis=-1).reshape(-1,1,2)
            uv=cv2.perspectiveTransform(points,np.linalg.inv(transform))
            if scale_map is not None:
                uv_map = uv.reshape(height,width,2).astype(np.float32)
                mask[:] = 0
                mask[roi] = cv2.remap(valid.astype(np.uint8),uv_map,None,cv2.INTER_NEAREST)*self.region[roi]
            norm=cv2.perspectiveTransform(uv,np.linalg.inv(K))
            if model=='fisheye':
                src=cv2.fisheye.distortPoints(norm.astype(np.float64),K,D)
            else:
                xyz=np.concatenate([norm.reshape(-1,2),np.ones((len(norm),1))],axis=1)
                src=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,D)[0]
            maps=src.reshape(height,width,2).astype(np.float32)
            warp=np.zeros_like(self.best)
            warp[y:y+height,x:x+width]=cv2.remap(raw,maps[...,0],maps[...,1],cv2.INTER_CUBIC)
        else:
            warp=cv2.warpPerspective(image, transform, self.size, flags=cv2.INTER_CUBIC)
        if diagnose and self.previous is not None:
            old, oldmask, oldpos = self.previous
            e = overlap_error(old, warp, (oldmask & mask) > 0)
            e.update(position=list(position), previous_position=list(oldpos))
            if e['valid']:
                # Content shift is the negative of actual-minus-commanded camera travel.
                e['visual_minus_command_mm'] = (-np.array(e['shift_px'])/self.ppm).tolist()
            self.errors.append(e)
        # Prefer optical centre in source coordinates, not distance to clipped output boundary.
        sh,sw=image.shape[:2]
        sy,sx=np.indices((sh,sw),dtype=np.float32)
        quality=np.exp(-2*((sx-(sw-1)/2)/(sw/2))**2-2*((sy-(sh-1)/2)/(sh/2))**2)*valid
        if scale_map is None:
            w=cv2.warpPerspective(quality,transform,self.size)*mask
        else:
            w=np.zeros_like(self.weight)
            w[roi]=cv2.remap(quality,uv_map,None,cv2.INTER_LINEAR)*mask[roi]
        choose=w>self.weight
        if self.blend_lighting:
            x, y, width, height = cv2.boundingRect(mask)
            if width and height:
                band = np.s_[y:y+height,x:x+width]
                covered = mask[band].astype(np.float32)
                sigma = max(2., .9*self.ppm)
                blur = lambda a: cv2.GaussianBlur(a,(0,0),sigma)
                # Mix illumination across overlap; keep edges and texture from one frame.
                light = blur(warp[band]*covered[...,None])/np.maximum(blur(covered)[...,None],1e-6)
                distance = cv2.distanceTransform(np.pad(covered.astype(np.uint8),1),cv2.DIST_L2,3)[1:-1,1:-1]
                weight = np.square(w[band])*np.minimum(1., distance/(2*sigma))
                self.light_sum[band] += light*weight[...,None]
                self.light_weight[band] += weight
                selected = choose[band]
                self.source_light[band][selected] = np.rint(light[selected]).clip(0,255).astype(np.uint8)
        self.best[choose]=warp[choose]
        if owner is not None:
            owner[choose] = tile_id
        self.weight=np.maximum(self.weight,w)
        self.previous = (warp, mask, list(position))

    def image(self):
        if self.blend_lighting:
            light = self.light_sum/np.maximum(self.light_weight[...,None],1e-9)
            return np.rint(self.best.astype(np.float32)-self.source_light+light).clip(0,255).astype(np.uint8)
        return self.best.copy()

    def xy(self, pixel):
        return self.origin + np.asarray(pixel)/self.ppm
