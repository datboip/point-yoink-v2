# SPDX-License-Identifier: AGPL-3.0-or-later
"""Evidence-backed starting settings and a read-only surface hint."""
import json
from pathlib import Path
import cv2
import numpy as np
from track import REGISTRATION_MIN_POINTS


def profiles(mode):return json.loads(Path(__file__).with_name('surface_profiles.json').read_text())[mode]

def preset(mode,name):return dict(profiles(mode)['presets'][name])

def hint(depth,ir,scene,color_missing=None):
    mask=scene.get('mask')
    if scene.get('source')!='part' or mask is None or not mask.any():return dict(kind='unknown',text='Show the part above the bed',dark_pixels=0,shiny_pixels=0)
    roi=np.zeros(depth.shape,np.uint8);contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(roi,contours,-1,1,cv2.FILLED)
    # Percentiles come from this observed part, with the actual sensor maximum
    # as the saturation threshold. No gain-specific brightness is invented.
    low=np.percentile(ir[mask],25);dark=(depth==0)&roi.astype(bool)&(ir<=low);bright=(ir==np.iinfo(ir.dtype).max)&roi.astype(bool)
    a,b=int(dark.sum()),int(bright.sum())
    if b>=REGISTRATION_MIN_POINTS:return dict(kind='shiny',text='Shiny spots: try Shiny or spray',dark_pixels=a,shiny_pixels=b)
    if a>=REGISTRATION_MIN_POINTS:return dict(kind='dark',text='Dark surface: try Dark or a bit of scanning spray',dark_pixels=a,shiny_pixels=b)
    return dict(kind='good',text='Good surface',dark_pixels=a,shiny_pixels=b)


def safe_range(scene,mode,current,margin=None):
    """The part's span plus a noise margin, or None when the part's depth is too sparse to say (a shiny part): the caller
    then falls back to the depth histogram. Expanding from the current Range on sparse evidence, as this did before,
    ratcheted Auto Range out to 0-1000 mm on shiny parts and never let it back in."""
    p=profiles(mode);noise=max(scene.get('noise_mm',0),p['noise_sigma_mm']);padding=max(3*noise,margin or 0)
    minimum=6*noise;span=scene.get('range_mm')
    if span is None:return None
    lo,hi=span;lo-=padding;hi+=padding;center=(lo+hi)/2
    lo=min(lo,center-minimum/2);hi=max(hi,center+minimum/2)
    from app2.scan.range_guide import measured_profile
    sparse=scene.get('fill',0)<measured_profile(mode)['fill_quartiles'][0]
    if sparse:return None
    from app2.scan.tracking import LIVE_MAX_DEPTH_MM
    return max(0,lo),min(LIVE_MAX_DEPTH_MM,hi)
