(function setupCalculator() {
  const form = document.getElementById('calc-form');
  if (!form) return;
  const iEl = document.getElementById('input-current');
  const rEl = document.getElementById('input-resistance');
  const pEl = document.getElementById('output-power');

  function computePowerKw(iKiloAmps, rMicroOhms) {
    const iAmps = iKiloAmps * 1_000; // kA -> A
    const rOhms = rMicroOhms * 1e-6; // micro-ohm -> ohm
    const pWatts = iAmps * iAmps * rOhms; // I^2 R
    const pKw = pWatts / 1000;
    return pKw;
  }

  function formatNumber(value) {
    return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : '';
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    const i = parseFloat(iEl.value);
    const r = parseFloat(rEl.value);
    if (!Number.isFinite(i) || !Number.isFinite(r) || i < 0 || r < 0) {
      pEl.value = '';
      return;
    }
    const p = computePowerKw(i, r);
    pEl.value = formatNumber(p);
  });
})();


