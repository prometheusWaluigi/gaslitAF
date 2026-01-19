const defaultParams = {
  ax: 0.35,
  bxy: 0.45,
  stressImpact: 0.6,
  ay: 0.4,
  kf: 0.3,
  byx: 0.35,
  dy: 0.25,
  af: 0.5,
  bfx: 0.2,
  df: 0.3,
};

const defaultState = {
  x: 0.6,
  y: 0.3,
  f: 0.2,
};

function createStressSchedule({ baseline = 0.2, events = [] } = {}) {
  return (t) => {
    const eventLoad = events.reduce((acc, event) => {
      if (t >= event.start && t <= event.end) {
        return acc + event.magnitude;
      }
      return acc;
    }, 0);

    return Math.max(0, baseline + eventLoad);
  };
}

function clampState(state) {
  return {
    x: Math.max(0, state.x),
    y: Math.max(0, state.y),
    f: Math.max(0, state.f),
  };
}

function derivative(state, t, params, stressLoad) {
  const safeState = clampState(state);
  const { x, y, f } = safeState;
  const load = stressLoad(t);

  const dx = params.ax * (1 - x) - params.bxy * y - params.stressImpact * load;
  const dy = params.ay * (load + params.kf * f) - params.byx * x - params.dy * y;
  const df = params.af * y - params.bfx * x - params.df * f;

  return { dx, dy, df, load };
}

function rk4Step(state, t, dt, params, stressLoad) {
  const k1 = derivative(state, t, params, stressLoad);
  const k2 = derivative(
    {
      x: state.x + (dt / 2) * k1.dx,
      y: state.y + (dt / 2) * k1.dy,
      f: state.f + (dt / 2) * k1.df,
    },
    t + dt / 2,
    params,
    stressLoad
  );
  const k3 = derivative(
    {
      x: state.x + (dt / 2) * k2.dx,
      y: state.y + (dt / 2) * k2.dy,
      f: state.f + (dt / 2) * k2.df,
    },
    t + dt / 2,
    params,
    stressLoad
  );
  const k4 = derivative(
    {
      x: state.x + dt * k3.dx,
      y: state.y + dt * k3.dy,
      f: state.f + dt * k3.df,
    },
    t + dt,
    params,
    stressLoad
  );

  return clampState({
    x: state.x + (dt / 6) * (k1.dx + 2 * k2.dx + 2 * k3.dx + k4.dx),
    y: state.y + (dt / 6) * (k1.dy + 2 * k2.dy + 2 * k3.dy + k4.dy),
    f: state.f + (dt / 6) * (k1.df + 2 * k2.df + 2 * k3.df + k4.df),
    load: k1.load,
  });
}

function simulateMinimalOde({
  initialState = defaultState,
  params = defaultParams,
  stressSchedule = createStressSchedule(),
  dt = 0.01,
  steps = 1000,
} = {}) {
  const states = [];
  const times = [];
  const stressLoads = [];

  let state = clampState({ ...initialState });

  for (let i = 0; i < steps; i += 1) {
    const t = i * dt;
    const next = rk4Step(state, t, dt, params, stressSchedule);

    times.push(t);
    stressLoads.push(next.load);
    states.push({ x: next.x, y: next.y, f: next.f });

    state = next;
  }

  return {
    times,
    stressLoads,
    states,
    final: { ...state },
  };
}

module.exports = {
  createStressSchedule,
  defaultParams,
  defaultState,
  simulateMinimalOde,
};
