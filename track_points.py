import cv2
import sdl2
import sdl2.ext
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

'''
Tracking feature points in a video, and highlight its movement

It detects features points using opencv2's goodFeaturesToTrack, this returns us
the point of the feature, but we also need its descpritor, which is like a
fingerprint of the image. To be able to match it with the same feature in the
next frame (in case the feature gets detected as feature again).

When we get a match of two features between two frames we store it in a feature
chain, this chain will be drawn on the image to show the movement of a feature.
We only accept features if they are not too far off, meaning we filter outliers
using RANSAC algorithm and a `FundamentalMatrixTransform` fit.
(FundamentalMatrix: ... relates corresponding points between a pair of
uncalibrated images.) (p.238)

Yea and thats it, the question is now can we use the drawn chains to calculate
the speed of the moving camera?

Resources:
* RANSAC, https://www.youtube.com/watch?v=9D5rrtCC_E0&ab_channel=CyrillStachniss
* Matrices, https://www.youtube.com/watch?v=auhpPoAqprk&ab_channel=CyrillStachniss
'''

W = 640
H = 480

sdl2.ext.init()
window = sdl2.ext.Window("Tracking Feature Points", size=(W,H), position=(0,0))
window.show()

class FeatureExctractor(object):
  def __init__(self):
    self.orb = cv2.ORB_create(100)
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True)
    self.last = None
    self.d_chain = {}

  def extract(self, img):

    # rely on opencv2's goodFeaturesToTrack
    features = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                                      3000, qualityLevel=0.01, minDistance=3)

    # descriptor extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
    kps, des = self.orb.compute(img, kps)

    if self.last is None:
      # on the first frame we dont have previous feature points
      self.last = {'kps': kps, 'des': des}
      return [], None

    # match features between frames
    latest_features = []
    matches = self.bf.knnMatch(des, self.last['des'], k=2)
    for m,n in matches:
      if m.distance >= 0.75*n.distance:
        continue

      new_p = kps[m.queryIdx].pt
      old_p = self.last['kps'][m.trainIdx].pt
      latest_features.append((new_p, old_p))

      # append matches to d_chain, last feature is the dict key
      if old_p not in self.d_chain:
        self.d_chain[new_p] = [old_p, new_p]
      else:
        c_chain = self.d_chain.pop(old_p)
        c_chain.append(new_p)
        self.d_chain[new_p] = c_chain

    # filter via fundamental matrix and ransac algorithm
    if len(latest_features) > 0:
      latest_features = np.array(latest_features)
      model, inliers = ransac((latest_features[:, 0], latest_features[:, 1]),
                              FundamentalMatrixTransform,
                              min_samples=8,
                              residual_threshold=1,
                              max_trials=100)
      latest_features = latest_features[inliers]

    # update last detected features
    self.last = {'kps': kps, 'des': des}

    # ret latest features to know which to draw (only prev points are needed)
    return latest_features[:,0], self.d_chain

fe = FeatureExctractor()


def process_frame(img):
  img = cv2.resize(img, (W,H))
  matches, d_chain = fe.extract(img)

  for p_pt in matches:

    # if we dont have the previous point in the d_chain, we dont draw
    if (p_pt[0], p_pt[1]) not in d_chain:
      continue

    # draw complete chain, as we have it
    draw_chain = d_chain[(p_pt[0], p_pt[1])]
    for i in range(len(draw_chain)-1):
      # round point cords, floats cant be drawn obviously
      u1,v1 = map(lambda x: int(round(x)), draw_chain[i])
      u2,v2 = map(lambda x: int(round(x)), draw_chain[i+1])

      # color = (blue, green, red)
      cv2.line(img, (u1,v1), (u2,v2), color=(0,255,0))
      # mark the point where the lines connect
      cv2.circle(img, (u1,v1), color=(0,0,255), radius=0)


  events = sdl2.ext.get_events()
  for event in events:
    if event.type == sdl2.SDL_QUIT:
      exit(0)

  surf = sdl2.ext.pixels3d(window.get_surface())
  surf[:, :, 0:3] = img.swapaxes(0,1)
  window.refresh()
  #input()


if __name__ == "__main__":
  cap = cv2.VideoCapture("videos/dashcam.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
