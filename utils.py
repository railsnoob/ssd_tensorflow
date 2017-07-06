import numpy as np

def print_stats2(the_y,name):
    indices = np.where(the_y != 0.0)[1]
    print(name+" indices=",indices)
    print(name," != 0.0", the_y[0][indices]  )
    print(name+" indices", indices.shape )


def iou(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2):
    # sum of areas
    xL = max(lx1,lx2)
    yL = max(ly1,ly2)
    xR = min(rx1,rx2)
    yR = min(ry1,ry2)

    intersection = 0
    if xR < xL or yR < yL:
        intersection = 0
    else:
        intersection = (xR - xL) * (yR - yL)

    union = (rx1-lx1)*(ry1-ly1) * 1.0 + (rx2-lx2)*(ry2-ly2)*1.0 - 1.0*intersection
        
    if(intersection*1.0/union > 0.3):
        print("  dbox=",lx1,ly1,rx1,ry1,"gt=",lx2,ly2,rx2,ry2);
        print("  intersection = ",intersection,"Union=",union,"IOUU=",intersection*1.0/union)
    
    return intersection*1.0/union


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")
