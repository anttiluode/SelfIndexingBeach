"""Self-indexing beach: a controlled, discrete wave / writable-bed experiment.

Not a physical sediment transport model or a biophysical neuron.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:
    n: int = 36
    c0: float = 0.14
    beta: float = 2.3
    friction: float = 0.14
    leak: float = 0.018
    write_eta: float = 0.14
    write_threshold: float = 0.0003
    max_sand: float = 0.95
    write_steps: int = 100
    read_steps: int = 120
    read_amp: float = 0.014
    write_amp: float = 0.24

class Beach:
    """Periodic ring of wave cells. Link sand changes coupling and hence travel times."""
    def __init__(self, config=Config(), sand=None, plastic=True):
        self.p = config
        self.sand = (np.zeros(config.n) if sand is None else np.asarray(sand, dtype=float).copy())
        self.water = np.zeros(config.n)
        self.flux = np.zeros(config.n)
        self.plastic = plastic
        self.written = 0.0

    def reset_water(self):
        self.water.fill(0)
        self.flux.fill(0)

    def step(self, source=0, impulse=0.0, allow_write=False):
        p=self.p
        self.water[source] += impulse
        gradient=self.water - np.roll(self.water,-1)
        self.flux=(1-p.friction)*self.flux + p.c0*np.exp(p.beta*self.sand)*gradient
        self.water = (self.water-self.flux+np.roll(self.flux,1))*(1-p.leak)
        if allow_write and self.plastic:
            dw=p.write_eta*np.maximum(gradient**2-p.write_threshold,0)*(1-self.sand/p.max_sand)
            self.sand=np.minimum(p.max_sand,self.sand+dw)
            self.written += float(dw.sum())
        return self.water

    def write(self, event):
        """Sine-packet A/B; fast wave reset makes sand the only inter-event memory."""
        assert event in ('A','B')
        self.reset_water()
        pos = 5 if event=='A' else 20
        frequency=0.043 if event=='A' else 0.079
        for t in range(self.p.write_steps):
            envelope=np.sin(np.pi*(t+1)/(self.p.write_steps+1))**2
            self.step(pos,self.p.write_amp*envelope*np.sin(2*np.pi*frequency*t), True)
        self.reset_water()

    def echo(self, freq, phase=0.0):
        """A nonwriting weak read; fixed shared port but varied frequencies/phases."""
        p=self.p
        self.reset_water()
        recordings=[]
        for t in range(p.read_steps):
            env=np.sin(np.pi*(t+1)/49)**2 if t<48 else 0.0
            wave=p.read_amp*env*np.sin(2*np.pi*freq*t+phase)
            state=self.step(0,wave,False)
            recordings.append(state[[8,17,26,33]].copy())
        self.reset_water()
        return np.asarray(recordings)

    def fingerprint(self, probes=((0.04,0),(0.07,0),(0.11,0),(0.07,np.pi/2)), bins=12):
        """Low-bandwidth readout: time-window averages at four listening ports."""
        parts=[]
        for freq,phase in probes:
            e=self.echo(freq,phase)
            parts.append(e.reshape(bins,-1,4).mean(axis=1).ravel())
        return np.concatenate(parts)

    def copy(self, plastic=None):
        return Beach(self.p,self.sand,self.plastic if plastic is None else plastic)

def sensitivity_matrix(beach, probes=((0.07,0),), perturb=1e-4):
    """Finite-difference Jacobian of echoes w.r.t. the material links.

    This is NOT a Hankel matrix and its numerical rank is not the state dimension.
    """
    base=beach.fingerprint(probes)
    jac=np.empty((base.size,beach.p.n))
    for j in range(beach.p.n):
        alt=beach.copy()
        alt.sand[j]+=perturb
        jac[:,j]=(alt.fingerprint(probes)-base)/perturb
    return jac
