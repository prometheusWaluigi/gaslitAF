const {
  createStressSchedule,
  defaultParams,
  simulateMinimalOde,
} = require('../src/simulations/minimalOde');

const runSimulation = (overrides = {}) =>
  simulateMinimalOde({
    dt: 0.01,
    steps: 2000,
    ...overrides,
  });

const expectClose = (value, expected, tolerance = 1e-6) => {
  expect(Math.abs(value - expected)).toBeLessThanOrEqual(tolerance);
};

describe('minimal ODE simulation', () => {
  test('returns stable output lengths and deterministic results', () => {
    const stressSchedule = createStressSchedule({ baseline: 0.3 });
    const first = runSimulation({ stressSchedule });
    const second = runSimulation({ stressSchedule });

    expect(first.times).toHaveLength(2000);
    expect(first.states).toHaveLength(2000);
    expect(first.stressLoads).toHaveLength(2000);

    const sampleIndex = 1500;
    expectClose(first.states[sampleIndex].x, second.states[sampleIndex].x);
    expectClose(first.states[sampleIndex].y, second.states[sampleIndex].y);
    expectClose(first.states[sampleIndex].f, second.states[sampleIndex].f);
  });

  test('low stress converges toward higher autonomic capacity', () => {
    const stressSchedule = createStressSchedule({ baseline: 0.1 });
    const result = runSimulation({ stressSchedule });
    const final = result.final;

    expect(final.x).toBeGreaterThan(0.65);
    expect(final.y).toBeLessThan(0.25);
    expect(final.f).toBeLessThan(0.25);
  });

  test('high stress drives higher fatigue and lower autonomic capacity', () => {
    const stressSchedule = createStressSchedule({
      baseline: 0.6,
      events: [{ start: 8, end: 12, magnitude: 0.6 }],
    });
    const result = runSimulation({ stressSchedule });
    const final = result.final;

    expect(final.x).toBeLessThan(0.4);
    expect(final.f).toBeGreaterThan(0.35);
  });

  test('stress schedule aggregates events on top of baseline', () => {
    const schedule = createStressSchedule({
      baseline: 0.2,
      events: [
        { start: 1, end: 2, magnitude: 0.4 },
        { start: 1.5, end: 1.7, magnitude: 0.2 },
      ],
    });

    expectClose(schedule(0.5), 0.2);
    expectClose(schedule(1.2), 0.6);
    expectClose(schedule(1.6), 0.8);
  });

  test('parameter overrides shift recovery dynamics', () => {
    const stressSchedule = createStressSchedule({ baseline: 0.1 });
    const boostedRecovery = runSimulation({
      stressSchedule,
      params: {
        ...defaultParams,
        ax: 0.6,
        bxy: 0.3,
      },
    });
    const baseline = runSimulation({ stressSchedule });

    expect(boostedRecovery.final.x).toBeGreaterThan(baseline.final.x);
  });
});
