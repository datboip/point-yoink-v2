# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plain descriptions of all active build nodes, without a GUI dependency."""
NAMES={'side.fuse':'Fusion','side.platter':'Cut base','side.keep':'Isolation','lineup.pairs':'Line-up pairs',
       'lineup.place':'Place sides','lineup.refine':'Refine line-up','fused':'Fusion','meshed':'Mesh','report':'Coverage'}

def headline(nodes,words=None):
    live=[(key,node) for key,node in (nodes or {}).items() if node.get('state')=='start']
    priority={'side.fuse':0,'side.platter':1,'side.keep':2}
    live.sort(key=lambda x:(priority.get(x[0].split('@')[0],3),x[1].get('n') or 0,x[0]))
    groups={}
    for key,node in live:
        nid=node.get('id',key);kind=nid.split('@')[0];n=node.get('n')
        label=(words(nid) if words else '') or NAMES.get(kind,nid)
        if kind.startswith('side.') and n:
            action=label.split(': ',1)[-1] if ': ' in label else NAMES.get(kind,label)
            groups.setdefault(('side',action),[]).append(n)
        else:groups.setdefault(('work',label),[])
    out=[]
    for (kind,label),nums in groups.items():
        if kind=='side':
            values=[str(x) for x in sorted(set(nums))]
            joined=values[0] if len(values)==1 else ', '.join(values[:-1])+' and '+values[-1]
            label=('Side ' if len(values)==1 else 'Sides ')+joined+': '+label
        out.append(label)
    return ', '.join(out)
