"""Closed-loop hypothesis test: can a nonwriting echo predict how the bed's responses change?"""
import argparse
import json
from pathlib import Path
import numpy as np
from beach import Beach, Config, sensitivity_matrix

PROBES = ((0.04, 0.), (0.07, 0.), (0.11, 0.), (0.07, np.pi/2))


def sample_episodes(n_episodes, steps, seed, plastic=True):
    rng = np.random.default_rng(seed)
    starts, events, changes, beds = [], [], [], []
    for _ in range(n_episodes):
        # Different initial beds, but a common known physical rule.
        rough = rng.uniform(0, 0.11, size=36)
        sand = (rough+np.roll(rough,1)+np.roll(rough,-1))/3
        beach = Beach(sand=sand, plastic=plastic)
        for j in range(steps):
            before = beach.fingerprint(PROBES)
            event = int(rng.integers(0,2))
            beach.write('A' if event==0 else 'B')
            after = beach.fingerprint(PROBES)
            starts.append(before)
            events.append(event)
            changes.append(after-before)
            beds.append(beach.sand.copy())
    return np.array(starts),np.array(events),np.array(changes),np.array(beds)


def design(x, events, mode='read', mu=None, scale=None):
    # Prepare feature scaling ONLY using train data. Same transform used at test time.
    if mu is None: mu=x.mean(0)
    if scale is None: scale=np.maximum(x.std(0), 1e-8)
    z=np.clip((x-mu)/scale, -20, 20)
    a=(events==0).astype(float)[:,None]
    b=1-a
    if mode=='event': return np.hstack([a,b]),mu,scale
    if mode=='read': return np.hstack([a,b,z*a,z*b]),mu,scale
    if mode=='no_event':return np.hstack([np.ones_like(a),z]),mu,scale
    raise ValueError(mode)


def ridge(X,Y,alpha=1e-3):
    # SVD least squares; regularize singular directions and avoid inverting X'X.
    U,s,Vt = np.linalg.svd(X,full_matrices=False)
    return (Vt.T*(s/(s*s+alpha))) @ (U.T@Y)


def score(target,pred):
    mse=np.mean((target-pred)**2)
    denom=np.mean(target**2)
    return {'mse':float(mse),'relative_change_error':float(mse/denom) if denom else 0}


def run(train_episodes=60,test_episodes=20,steps=6,seed=13,plot=True):
    x,e,y,_=sample_episodes(train_episodes,steps,seed)
    tx,te,ty,_=sample_episodes(test_episodes,steps,seed+10000)
    results={}
    for mode in ('event','read','no_event'):
        X,mu,std=design(x,e,mode)
        TX,_,_=design(tx,te,mode,mu,std)
        W=ridge(X,y,alpha=0.1)
        results[mode]=score(ty,TX@W)
    results['no_change']=score(ty,np.zeros_like(ty))
    # Control: violate link between each observed bed signature and its next write.
    sh=np.random.default_rng(seed+20000).permutation(len(x))
    X,mu,std=design(x[sh],e,'read')
    TX,_,_=design(tx,te,'read',mu,std)
    W=ridge(X,y,alpha=0.1)
    results['shuffled_read']=score(ty,TX@W)
    results['read_beats_event']=(results['event']['mse']-results['read']['mse'])/results['event']['mse']
    # Frozen bed: read-only and write both should preserve the bed (and echoes).
    f=Beach(plastic=False)
    sf=f.sand.copy();f0=f.fingerprint(PROBES);f.write('A');f1=f.fingerprint(PROBES)
    results['frozen_max_sand_change']=float(np.max(np.abs(f.sand-sf)))
    results['frozen_max_echo_change']=float(np.max(np.abs(f1-f0)))
    # AB/BA symmetry challenge from the same virgin bed.
    ab=Beach();ba=Beach()
    ab.write('A');ab.write('B');ba.write('B');ba.write('A')
    results['ab_ba_sand_l2']=float(np.linalg.norm(ab.sand-ba.sand))
    results['ab_ba_echo_l2']=float(np.linalg.norm(ab.fingerprint(PROBES)-ba.fingerprint(PROBES)))
    # Local *material* observability; not a dynamical Hankel rank.
    J1 = sensitivity_matrix(ab, (PROBES[1],))
    J4 = sensitivity_matrix(ab, PROBES)
    sv1 = np.linalg.svd(J1, compute_uv=False)
    sv4 = np.linalg.svd(J4, compute_uv=False)
    threshold = 1e-4  # explicit fixed absolute gain threshold; measurement-dependent
    results['sensitivity'] = {
        'measurement_gain_threshold': threshold,
        'one_probe_directions_above_threshold': int(np.count_nonzero(sv1>threshold)),
        'four_probe_directions_above_threshold': int(np.count_nonzero(sv4>threshold)),
        'one_probe_singular_values': sv1.tolist(),
        'four_probe_singular_values': sv4.tolist(),
    }
    results['train_n']=int(len(x));results['test_n']=int(len(tx))
    results['seed']=seed
    if plot:
        import matplotlib.pyplot as plt
        plt.rcParams['svg.fonttype']='none'
        Path('results').mkdir(exist_ok=True)
        names=['no_change','event','shuffled_read','no_event','read']
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.bar(names,[results[n]['relative_change_error'] for n in names])
        ax.set_ylabel('MSE / mean(true change²); lower is better')
        ax.set_title('Predicting the next echo change after a known sand-writing event')
        ax.tick_params(axis='x',rotation=20)
        ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig('results/predictive_mirror.png',dpi=150);fig.savefig('results/predictive_mirror.svg')
        plt.close(fig)
        fig,axs=plt.subplots(2,1,figsize=(9,5.5),sharex=True)
        axs[0].plot(ab.sand,label='AB');axs[0].plot(ba.sand,label='BA',linestyle='--');axs[0].set_ylabel('Link sand');axs[0].legend()
        axs[1].plot(ab.sand-ba.sand);axs[1].set_ylabel('AB − BA');axs[1].set_xlabel('Ring link index');axs[1].grid(alpha=.2)
        fig.suptitle('The writing order affects the bed, but only weakly')
        fig.tight_layout();fig.savefig('results/order_hidden.png',dpi=150);fig.savefig('results/order_hidden.svg')
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.semilogy(np.arange(1,len(sv1)+1),sv1,marker='o',markersize=3,label='One frequency')
        ax.semilogy(np.arange(1,len(sv4)+1),sv4,marker='o',markersize=3,label='Four frequency/phase probes')
        ax.axhline(threshold,linestyle='--',linewidth=1,label='Chosen gain threshold')
        ax.set_xlabel('Singular-value index');ax.set_ylabel('Echo sensitivity per unit sand')
        ax.set_title('Local material observability, not Hankel rank')
        ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig('results/observability.png',dpi=150);fig.savefig('results/observability.svg')
        plt.close(fig)
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',type=int,default=60)
    ap.add_argument('--test',type=int,default=20)
    ap.add_argument('--steps',type=int,default=6)
    ap.add_argument('--seed',type=int,default=13)
    ap.add_argument('--no-plot',action='store_true')
    args=ap.parse_args()
    res=run(args.train,args.test,args.steps,args.seed,plot=not args.no_plot)
    Path('results').mkdir(exist_ok=True)
    Path('results/summary.json').write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(res,indent=2))
