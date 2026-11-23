"""Height consensus and nearest-pixel checks between reference depth maps."""
import numpy as np


def consensus(candidates, tolerance):
    """Return memberships of local height groups, keeping ties for the caller."""
    order = np.argsort(candidates, axis=0)
    z = np.take_along_axis(candidates, order, axis=0)
    with np.errstate(invalid="ignore"):
        return (z[None] >= z[:, None]) & (z[None] - z[:, None] <= tolerance)


def robust_consensus(layers, points, cameras, offsets, normal, distance, tolerance, height, confirmed):
    shape = height.shape
    z = layers.reshape(len(layers), -1)
    xyz = points.reshape(len(layers), -1, 3)
    votes = np.isfinite(z).sum(axis=0)
    target = np.flatnonzero((votes > 0) & ~confirmed.ravel())
    result = height.copy().ravel()
    status = np.zeros(votes.size, np.uint8)
    status[confirmed.ravel()] = 1
    count = np.where(confirmed.ravel(),votes,0)
    discarded = np.zeros_like(count)

    for start in range(0, len(target), 8192):
        columns = target[start:start+8192]
        values = z[:,columns]
        total = votes[columns]
        memberships = consensus(values,tolerance)
        sizes = memberships.sum(axis=1)
        winner = np.argmax(sizes,axis=0)
        size = sizes.max(axis=0)
        best = memberships[winner,:,np.arange(len(columns))].T
        tied = ((sizes == size) & np.any(memberships != best[None],axis=1)).sum(axis=0) > 0
        group = (size >= 3) & (2*size > total) & ~tied
        for k in np.flatnonzero(group):
            selected = np.sort(values[:,k])[best[:,k]]
            result[columns[k]] = np.median(selected)
        status[columns[group]] = 2
        count[columns[group]] = size[group]
        discarded[columns[group]] = total[group] - size[group]

        # Project each original sample, not a cell centre that could move across a sharp edge.
        unresolved = columns[~group]
        valid_source = np.isfinite(z[:,unresolved])
        supported = valid_source.astype(np.uint16)
        contradicted = np.zeros_like(supported)
        candidate_height = distance - xyz[:,unresolved] @ normal
        for view, (K, image) in enumerate(cameras):
            relative = xyz[:,unresolved] - offsets[view]
            projected = relative @ K.T
            front = np.isfinite(projected[...,2]) & (projected[...,2] > 0)
            front[view] = False
            u = np.zeros(front.shape, int)
            v = np.zeros_like(u)
            u[front],v[front] = np.rint(projected[front,:2]/projected[front,2,None]).astype(int).T
            inside = front & (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
            check = image[v[inside],u[inside]]
            delta = check - candidate_height[inside]
            agree = np.abs(delta) <= tolerance
            supported[inside] += agree
            # A nearer return occludes the tested point. A farther return contradicts it.
            contradicted[inside] += delta < -tolerance

        for k in np.flatnonzero(valid_source.any(axis=0)):
            valid = valid_source[:,k]
            good = valid & (supported[:,k] >= 3) & (supported[:,k] > contradicted[:,k])
            if not good.any():
                continue
            hypotheses = z[good,unresolved[k]]
            if np.ptp(hypotheses) > tolerance:
                continue
            indices = np.flatnonzero(good)
            source = indices[np.argmax(supported[indices,k])]
            # Preserve the original measured hypothesis. Other views certify its visibility and height.
            result[unresolved[k]] = z[source,unresolved[k]]
            status[unresolved[k]] = 3
            count[unresolved[k]] = supported[source,k]
            discarded[unresolved[k]] = np.count_nonzero(
                valid & (np.abs(z[:,unresolved[k]] - z[source,unresolved[k]]) > tolerance))
    return result.reshape(shape), status.reshape(shape), count.reshape(shape), discarded.reshape(shape)
