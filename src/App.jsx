import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowRight, BarChart3, Check, Database, FileSpreadsheet, Info, Pause, Play, RefreshCw, ShieldCheck, Sparkles, UploadCloud, Wand2, X } from 'lucide-react';
import { getJson, login, logout, formatPercent } from './lib/api.js';

const SAMPLE_CSV = `amount,device_age_days,velocity_10m,merchant_category,location_new,is_fraud
42,930,1,groceries,0,0
58,730,1,groceries,0,0
76,410,2,restaurants,0,0
91,680,1,fuel,0,0
112,520,2,travel,0,0
134,270,2,groceries,0,0
188,180,3,travel,1,0
220,120,3,electronics,1,1
340,80,4,electronics,1,1
520,24,5,electronics,1,1
780,12,6,electronics,1,1
1160,4,8,electronics,1,1
64,820,1,restaurants,0,0
97,310,1,fuel,0,0
155,190,2,travel,0,0
290,95,4,electronics,1,1
430,40,5,electronics,1,1
84,700,1,groceries,0,0
128,260,2,restaurants,0,0
650,18,7,electronics,1,1`;

const ALGORITHM_OPTIONS = [
  { id: 'logistic_regression', label: 'Logistic regression', note: 'Custom implementation · linear, interpretable baseline' },
  { id: 'gaussian_nb', label: 'Gaussian Naive Bayes', note: 'Custom implementation · fast probabilistic model' },
  { id: 'decision_tree', label: 'Decision tree', note: 'Custom CART · rule-like nonlinear splits' },
  { id: 'random_forest', label: 'Random Forest', note: 'Custom bagged ensemble (17 trees)' },
  { id: 'gradient_boosted_stumps', label: 'Gradient-boosted stumps', note: 'Custom boosting approximation · not native XGBoost' },
];

function LoginScreen({ onLogin, busy, error }) {
  const [token, setToken] = useState('');
  return <main className="auth-gate"><section className="auth-card"><span className="brand-shield"><ShieldCheck size={18} /></span><span className="section-label">PROTECTED INVESTIGATOR WORKSPACE</span><h1>Sign in to <em>FraudShield Z.</em></h1><p>Enter your administrator or investigator access token. It is exchanged for an HttpOnly session cookie and is not saved in browser storage.</p><form onSubmit={(event) => { event.preventDefault(); onLogin(token, () => setToken('')); }}><label htmlFor="access-token">Access token</label><input id="access-token" type="password" autoComplete="current-password" value={token} onChange={(event) => setToken(event.target.value)} required autoFocus /><button className="train-button" disabled={busy || !token}>{busy ? 'Signing in…' : 'Sign in securely'} <ArrowRight size={15} /></button></form>{error && <div className="auth-error" role="alert">{error}</div>}<small>Access is protected by role-based server authorization. Never share tokens or enter them into an untrusted page.</small></section></main>;
}

function App() {
  const [authStatus, setAuthStatus] = useState(null);
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState('');
  const [step, setStep] = useState(1);
  const [fileName, setFileName] = useState('');
  const [datasetText, setDatasetText] = useState('');
  const [preview, setPreview] = useState(null);
  const [labelColumn, setLabelColumn] = useState('');
  const [timeColumn, setTimeColumn] = useState('');
  const [model, setModel] = useState(null);
  const [models, setModels] = useState([]);
  const [experiments, setExperiments] = useState(null);
  const [monitoring, setMonitoring] = useState(null);
  const [algorithms, setAlgorithms] = useState(ALGORITHM_OPTIONS.map((option) => option.id));
  const [costMatrix, setCostMatrix] = useState({ false_negative_cost: 100, false_positive_cost: 10, review_cost: 1, false_negative_use_amount: false });
  const setCost = (key, value) => setCostMatrix((current) => ({ ...current, [key]: key === 'false_negative_use_amount' ? Boolean(value) : Number(value) }));
  const [row, setRow] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [simulations, setSimulations] = useState([]);
  const [selectedSimulation, setSelectedSimulation] = useState(null);
  const [liveEvents, setLiveEvents] = useState([]);
  const [liveRunning, setLiveRunning] = useState(true);
  const streamRef = useRef(null);

  const setMessage = (message) => { setNotice(message); setError(''); };
  const fail = (message) => { setError(message); setNotice(''); };

  useEffect(() => {
    let current = true;
    getJson('/api/auth/session').then((status) => { if (current) setAuthStatus(status); })
      .catch(() => { if (current) setAuthStatus({ authenticated: false, mode: 'unavailable' }); });
    return () => { current = false; };
  }, []);

  useEffect(() => {
    const onUnauthorized = () => { setAuthStatus({ authenticated: false, mode: 'production-session' }); setAuthError('Your session expired or was revoked. Sign in again to continue.'); };
    window.addEventListener('fraudshield:unauthorized', onUnauthorized);
    return () => window.removeEventListener('fraudshield:unauthorized', onUnauthorized);
  }, []);

  const signIn = async (token, clearToken) => {
    setAuthBusy(true); setAuthError('');
    try { const result = await login(token); clearToken(); setAuthStatus(result); }
    catch (requestError) { clearToken(); setAuthError(requestError.message); }
    finally { setAuthBusy(false); }
  };
  const signOut = async () => {
    try { await logout(); } finally { setAuthStatus({ authenticated: false, mode: 'production-session' }); setStep(1); setFileName(''); setDatasetText(''); setPreview(null); setModel(null); setModels([]); setExperiments(null); setPrediction(null); setRow({}); }
  };

  useEffect(() => {
    if (!authStatus?.authenticated) return undefined;
    getJson('/api/scenarios').then((result) => {
      setSimulations(result.scenarios);
      setSelectedSimulation(result.scenarios[2]);
    }).catch(() => {});
  }, [authStatus?.authenticated]);

  useEffect(() => {
    if (!authStatus?.authenticated) { streamRef.current?.close(); return undefined; }
    if (!liveRunning) { streamRef.current?.close(); return undefined; }
    const source = new EventSource('/api/stream');
    streamRef.current = source;
    source.onmessage = (event) => {
      try { setLiveEvents((current) => [{ ...JSON.parse(event.data), received_at: new Date().toISOString() }, ...current].slice(0, 5)); }
      catch { /* keep the rail calm when an event is malformed */ }
    };
    source.onerror = () => source.close();
    return () => source.close();
  }, [liveRunning, authStatus?.authenticated]);

  const loadDataset = async (name, content) => {
    setBusy(true); setFileName(name); setDatasetText(content);
    try {
      const result = await getJson('/api/ml/preview', { method: 'POST', body: JSON.stringify({ content, labelColumn }) });
      setPreview(result.preview); setLabelColumn(result.preview.label_column); setTimeColumn(result.preview.time_column || ''); setStep(2); setMessage(`${result.preview.row_count.toLocaleString()} rows found. Ready to train.`);
    } catch (requestError) { fail(requestError.message); }
    finally { setBusy(false); }
  };

  const onFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => loadDataset(file.name, String(reader.result));
    reader.onerror = () => fail('Could not read that file. Try a CSV export.');
    reader.readAsText(file);
  };

  const train = async () => {
    setBusy(true);
    try {
      const result = await getJson('/api/ml/train', { method: 'POST', body: JSON.stringify({ content: datasetText, labelColumn, timeColumn: timeColumn || undefined, algorithms, costMatrix }) });
      setModel(result.model); setModels(result.models); setExperiments(result.experiments); setStep(3); setPrediction(null); setMessage(`${result.models.length} models trained and compared with temporal experiments. The winner was chosen on validation data only; the held-out test period was scored once for that model.`);
      const initialRow = Object.fromEntries((result.model.input_columns || result.model.feature_columns).map((column) => [column, '']));
      setRow(initialRow);
    } catch (requestError) { fail(requestError.message); }
    finally { setBusy(false); }
  };

  const predict = async () => {
    setBusy(true);
    try { setPrediction(await getJson('/api/ml/predict', { method: 'POST', body: JSON.stringify({ row, modelId: model?.model_id }) })); setMessage(`Prediction generated with ${model?.model_type || 'the selected model'}.`); }
    catch (requestError) { fail(requestError.message); }
    finally { setBusy(false); }
  };

  const chooseModel = (candidate) => { setModel(candidate); setPrediction(null); setMessage(`${candidate.model_type} selected for prediction.`); };
  const recordAudit = async (action) => { try { const result = await getJson('/api/ml/audit', { method: 'POST', body: JSON.stringify({ row, modelId: model?.model_id, action }) }); setMessage(`Audit ${result.entry.audit_id} recorded with explanation and final action.`); } catch (requestError) { fail(requestError.message); } };
  const runReplay = async () => { setBusy(true); try { setMonitoring(await getJson('/api/ml/replay', { method: 'POST', body: JSON.stringify({ content: datasetText, modelId: model?.model_id, labelColumn, timeColumn }) })); setMessage('Replay complete. Drift alerts are advisory and do not replace the active model.'); } catch (requestError) { fail(requestError.message); } finally { setBusy(false); } };
  const reset = () => { setStep(1); setFileName(''); setDatasetText(''); setPreview(null); setLabelColumn(''); setTimeColumn(''); setModel(null); setModels([]); setExperiments(null); setPrediction(null); setRow({}); setNotice(''); setError(''); };

  if (!authStatus) return <main className="auth-gate"><section className="auth-card"><span className="brand-shield"><ShieldCheck size={18} /></span><h1>Connecting securely…</h1><p>Checking the server session before loading the workbench.</p></section></main>;
  if (!authStatus.authenticated) return <LoginScreen onLogin={signIn} busy={authBusy} error={authError || (authStatus.mode === 'unavailable' ? 'The API is unavailable. Start the local server or contact the deployment administrator.' : '')} />;

  return <div className="ml-app">
    <header className="ml-nav"><a className="ml-brand" href="#top"><span className="brand-shield"><ShieldCheck size={17} /></span><span>FraudShield <b>Z</b></span></a><div className="nav-center"><span className="nav-context">ML WORKBENCH</span><span className="nav-divider" />{fileName ? <span className="nav-file"><FileSpreadsheet size={13} /> {fileName}</span> : <span className="nav-file muted">No dataset loaded</span>}</div><span className="auth-role">{authStatus.role}</span>{authStatus.mode === 'production-session' && <button className="nav-reset" onClick={signOut}>Sign out</button>}<button className="nav-reset" onClick={reset}><RefreshCw size={14} /> New model</button></header>
    <main id="top" className="ml-main">
      <section className="ml-hero"><div className="hero-tag"><i /> TRAIN YOUR OWN FRAUD MODEL</div><h1>From raw data<br /><em>to sharper decisions.</em></h1><p>Upload a labelled transaction dataset, train a transparent model, and test fraud predictions in a few guided steps.</p><div className="hero-statline"><span><strong>01</strong> upload a CSV</span><span><strong>02</strong> train a model</span><span><strong>03</strong> analyze risk</span></div></section>

      <section className="progress-track"><div className={`progress-step ${step >= 1 ? 'active' : ''}`}><span>01</span><div><strong>Upload</strong><small>Bring your data</small></div></div><div className="progress-line"><i style={{ width: `${step === 1 ? 0 : step === 2 ? 50 : 100}%` }} /></div><div className={`progress-step ${step >= 2 ? 'active' : ''}`}><span>02</span><div><strong>Train</strong><small>Learn the pattern</small></div></div><div className="progress-line"><i style={{ width: step >= 3 ? '100%' : '0%' }} /></div><div className={`progress-step ${step >= 3 ? 'active' : ''}`}><span>03</span><div><strong>Analyze</strong><small>Make a decision</small></div></div></section>

      {step === 1 && <section className="workspace-card upload-workspace"><div className="workspace-intro"><span className="section-label"><Database size={14} /> Start with a labelled dataset</span><h2>Give your model<br /><em>something to learn.</em></h2><p>Use almost any bank, card, payments, or transaction CSV. FraudShield detects common target names and lets you choose the label column yourself.</p><div className="feature-list"><span><Check size={13} /> Numeric and categorical columns</span><span><Check size={13} /> Flexible bank CSV headers</span><span><Check size={13} /> Compare five algorithms</span></div></div><div className="dropzone"><input id="dataset" type="file" accept=".csv,.txt" onChange={onFile} /><label htmlFor="dataset"><span className="upload-icon"><UploadCloud size={24} /></span><strong>{busy ? 'Reading dataset…' : 'Drop your dataset here'}</strong><small>or click to browse · CSV up to 10MB</small></label><button className="sample-button" onClick={() => loadDataset('fraudshield-demo.csv', SAMPLE_CSV)}><Wand2 size={14} /> Use sample dataset</button></div></section>}

      {step === 2 && <section className="workspace-card train-workspace"><div className="workspace-heading"><div><span className="section-label"><FileSpreadsheet size={14} /> Dataset ready</span><h2>{fileName}</h2><p>{preview?.row_count.toLocaleString()} rows · {preview?.columns.length} columns · {preview?.positive_rate !== null ? formatPercent(preview.positive_rate) : '—'} positive labels · chronological evaluation</p></div><button className="small-ghost" onClick={reset}><X size={14} /> Change dataset</button></div><div className="dataset-layout"><div className="table-preview"><div className="table-top"><span>Data preview</span><span>{preview?.columns.slice(0, 5).join(' · ')}{preview?.columns.length > 5 ? ' · …' : ''}</span></div><div className="data-table">{preview?.preview.map((item, index) => <div className="data-row" key={index}>{preview.columns.slice(0, 5).map((column) => <span key={column}>{String(item[column]).slice(0, 16)}</span>)}</div>)}</div></div><div className="train-controls"><label>Target column<select value={labelColumn} onChange={(event) => setLabelColumn(event.target.value)}>{preview?.columns.map((column) => <option key={column}>{column}</option>)}</select></label><label>Time column<select value={timeColumn} onChange={(event) => setTimeColumn(event.target.value)}><option value="">Use detected column / row order</option>{preview?.columns.filter((column) => column !== labelColumn).map((column) => <option key={column}>{column}</option>)}</select></label><div className="algorithm-picker"><span className="picker-label">Compare algorithms</span>{ALGORITHM_OPTIONS.map((option) => <label className="algorithm-option" key={option.id}><input type="checkbox" checked={algorithms.includes(option.id)} onChange={() => setAlgorithms((current) => current.includes(option.id) ? current.filter((item) => item !== option.id) : [...current, option.id])} /><span><b>{option.label}</b><small>{option.note}</small></span></label>)}</div><div className="cost-matrix"><span className="picker-label">Decision costs (any consistent currency unit)</span><label>Missed fraud<input type="number" min="0" value={costMatrix.false_negative_cost} onChange={(event) => setCost('false_negative_cost', event.target.value)} /></label><label>False alarm (friction)<input type="number" min="0" value={costMatrix.false_positive_cost} onChange={(event) => setCost('false_positive_cost', event.target.value)} /></label><label>Review / step-up action<input type="number" min="0" value={costMatrix.review_cost} onChange={(event) => setCost('review_cost', event.target.value)} /></label><label className="cost-check"><input type="checkbox" checked={costMatrix.false_negative_use_amount} onChange={(event) => setCost('false_negative_use_amount', event.target.checked)} /><span>Add transaction amount to missed-fraud cost (needs an amount column)</span></label></div><div className="training-note"><Info size={15} /><span>FraudShield sorts by time, trains on the first 60%, picks the winning model and the cost-minimising threshold on the next 20%, then scores the final 20% once for the winner. Derived behavioural features are rebuilt from past history only.</span></div><button className="train-button" onClick={train} disabled={busy || algorithms.length === 0}><Play size={15} /> {busy ? 'Training models…' : `Train ${algorithms.length} model${algorithms.length === 1 ? '' : 's'}`} <ArrowRight size={15} /></button></div></div></section>}

      {step === 3 && <><section className="comparison-panel workspace-card"><div className="workspace-heading"><div><span className="section-label"><BarChart3 size={14} /> Temporal model comparison</span><h2>Choose the pattern that fits.</h2><p>Models are ranked on validation data only. The held-out test period is scored once, for the selected model; other cards show validation figures and are not test results.</p></div><button className="small-ghost" onClick={() => setStep(2)}><BarChart3 size={14} /> Train again</button></div><div className="comparison-grid">{models.map((candidate) => <button key={candidate.model_id} className={`comparison-card ${candidate.model_id === model.model_id ? 'selected' : ''}`} onClick={() => chooseModel(candidate)}><span className="comparison-check">{candidate.model_id === model.model_id ? 'ACTIVE' : 'USE MODEL'}</span><strong>{candidate.model_type}</strong><small>{candidate.algorithm.replaceAll('_', ' ')}</small><div className="comparison-metrics"><span><b>{candidate.metrics.pr_auc === null ? '—' : candidate.metrics.pr_auc.toFixed(3)}</b> {candidate.metrics_source === 'test' ? 'test' : 'validation'} PR-AUC</span><span><b>{candidate.metrics.cost?.total_cost ?? '—'}</b> {candidate.metrics_source === 'test' ? 'test' : 'validation'} cost</span><span><b>{candidate.metrics.operational?.prediction_latency_ms.p95 ?? '—'}ms</b> p95 latency</span></div></button>)}</div></section>{experiments && <section className="experiment-panel workspace-card"><div className="workspace-heading"><div><span className="section-label"><BarChart3 size={14} /> Evaluation experiments</span><h2>What actually helped?</h2><p>Training variants and the anomaly detector stay separate from the supervised production candidate.</p></div></div><div className="experiment-grid">{experiments.training_variants.map((variant) => <div className="experiment-card" key={variant.variant}><span>{variant.variant.replaceAll('_', ' ')}</span><strong>{variant.metrics.pr_auc === null ? '—' : variant.metrics.pr_auc.toFixed(3)}</strong><small>PR-AUC · {variant.metrics.false_positives} FP · {variant.metrics.false_negatives} FN</small></div>)}<div className="experiment-card anomaly"><span>Isolation Forest</span><strong>{experiments.anomaly_detector.metrics.pr_auc === null ? '—' : experiments.anomaly_detector.metrics.pr_auc.toFixed(3)}</strong><small>separate anomaly PR-AUC · no hybrid blending</small></div>{experiments.graph_features && <div className="experiment-card graph"><span>Graph features</span><strong>{experiments.graph_features.graph_metrics.pr_auc === null ? '—' : experiments.graph_features.graph_metrics.pr_auc.toFixed(3)}</strong><small>{experiments.graph_features.feature_names.length} relationship features vs tabular baseline</small></div>}</div>{experiments.ablation && <div className="ablation-block"><h3>Controlled ablation · features × imbalance handling</h3><p className="ablation-note">{experiments.ablation.design} Exploratory test figures; nothing here influences model or threshold selection.</p>{experiments.ablation.reliability_warning && <div className="drift-banner alert">{experiments.ablation.reliability_warning}</div>}<div className="experiment-grid">{experiments.ablation.variants.map((variant) => <div className="experiment-card" key={variant.variant}><span>{variant.features} · {variant.imbalance_method}</span><strong>{variant.metrics.pr_auc === null ? '—' : variant.metrics.pr_auc.toFixed(3)}</strong><small>test PR-AUC · cost {variant.metrics.cost.total_cost} · {variant.synthetic_rows} synthetic rows</small></div>)}</div><div className="contrast-list">{Object.entries(experiments.ablation.contrasts).map(([name, item]) => <span key={name}>{name.replaceAll('_', ' ')}<b>{item.test_pr_auc_change === null ? '—' : `${item.test_pr_auc_change >= 0 ? '+' : ''}${item.test_pr_auc_change}`} PR-AUC · cost {item.test_cost_change >= 0 ? '+' : ''}{item.test_cost_change}</b></span>)}</div></div>}</section>}<section className="monitor-panel workspace-card"><div className="workspace-heading"><div><span className="section-label"><Activity size={14} /> Replay + drift monitor</span><h2>Replay before you retrain.</h2><p>Feed the current dataset through the real prediction endpoint. Alerts are advisory only.</p></div><button className="small-ghost" onClick={runReplay} disabled={busy}><Activity size={14} /> {busy ? 'Replaying…' : 'Run replay'}</button></div>{monitoring ? <><div className="monitor-metrics"><span><b>{monitoring.throughput_rows_per_second}</b> rows/sec</span><span><b>{monitoring.latency_ms.p95} ms</b> p95 latency</span><span><b>{monitoring.rows_replayed}</b> transactions</span><span className={monitoring.drift.alert ? 'alert' : ''}><b>{monitoring.drift.alert ? 'DRIFT ALERT' : 'STABLE'}</b> feature status</span></div><div className={`drift-banner ${monitoring.drift.alert ? 'alert' : 'stable'}`}>{monitoring.drift.message} No automatic retraining or model replacement is performed.</div></> : <div className="monitor-empty">No replay run yet. Run it after selecting a model to measure throughput, latency, later-period quality, and feature drift.</div>}</section><section className="analysis-layout"><div className="workspace-card model-result"><div className="workspace-heading"><div><span className="section-label"><Sparkles size={14} /> Selected model</span><h2>{model.model_type}</h2><p>{model.row_count.toLocaleString()} rows · {model.feature_columns.length} features · {model.model_version}</p></div><button className="small-ghost" onClick={() => setStep(2)}><BarChart3 size={14} /> Review data</button></div><div className="metric-grid"><div><span>{model.metrics.source === 'test' ? 'Test PR-AUC' : 'Validation PR-AUC'}</span><strong>{model.metrics.pr_auc === null ? '—' : model.metrics.pr_auc.toFixed(3)}</strong></div><div><span>Recall</span><strong>{formatPercent(model.metrics.recall)}</strong></div><div><span>False negatives</span><strong>{model.metrics.false_negatives}</strong></div><div><span>Cost-optimal threshold</span><strong>{model.metrics.threshold}</strong></div></div><div className="model-detail-grid"><div><h3>Operational evaluation</h3><div className="split-card"><span>{model.metrics.source === 'test' ? 'Test' : 'Validation'} prevalence <b>{formatPercent(model.metrics.prevalence)}</b></span><span>Total cost <b>{model.metrics.cost?.total_cost ?? '—'}</b></span><span>Saved vs approve-all <b>{model.metrics.cost?.savings_vs_approve_all ?? '—'}</b></span><span>Flagged workload <b>{formatPercent(model.metrics.operational?.false_positive_workload_rate ?? 0)}</b></span><span>p95 latency <b>{model.metrics.operational?.prediction_latency_ms.p95 ?? '—'} ms</b></span><span>Consistency <b>{formatPercent(model.metrics.operational?.decision_consistency_rate ?? 0)}</b></span><span>False positives <b>{model.metrics.false_positives}</b></span><span>False negatives <b>{model.metrics.false_negatives}</b></span></div><h3>Threshold outcomes</h3><div className="threshold-strip">{model.metrics.thresholds?.map((item) => <span key={item.threshold} className={item.selected ? 'selected' : ''}><b>{item.threshold}</b><small>{item.false_positives} FP · {item.false_negatives} FN · cost {item.total_cost}</small></span>)}</div></div><div><h3>Top signals · {model.importance_method}</h3>{model.top_features.length ? model.top_features.slice(0, 6).map((feature) => <div className="importance-row" key={feature.feature}><span>{feature.feature}</span><i><b style={{ width: `${Math.min(100, feature.importance * 100)}%` }} /></i><em title={feature.weight === null ? 'importance share' : `coefficient ${feature.weight}`}>{Math.round(feature.importance * 100)}%</em></div>) : <div className="monitor-empty">No importance signal available for this model.</div>}<div className="training-note"><Info size={15} /><span>Failure fallback: review_required, never automatic approval.</span></div></div></div></div><div className="workspace-card predictor-card"><div className="workspace-heading"><div><span className="section-label"><ShieldCheck size={14} /> Score a transaction</span><h2>What does the model see?</h2></div></div><p className="predictor-copy">Enter the raw fields of a new transaction (include the timestamp and customer if your data has them). Behavioural and relationship features are rebuilt automatically from stored history; you do not enter them.</p><div className="predict-form">{(model.input_columns || model.feature_columns).map((column) => <label key={column}><span>{column}</span><input value={row[column] ?? ''} placeholder="value" onChange={(event) => setRow((current) => ({ ...current, [column]: event.target.value }))} /></label>)}</div><button className="predict-button" onClick={predict} disabled={busy}><ShieldCheck size={15} /> {busy ? 'Scoring…' : 'Run prediction'} <ArrowRight size={15} /></button>{prediction && <div className={`prediction-result ${prediction.prediction === 'fraud' ? 'fraud' : 'legitimate'}`}><div><span>Prediction · {prediction.action === 'review_or_step_up' ? 'review / step-up' : 'approve'}</span><strong>{prediction.prediction === 'fraud' ? 'Fraud signal' : 'Likely legitimate'}</strong></div><div className="prediction-score"><span>Risk score</span><b>{Math.round(prediction.risk_score * 100)}<small>/100</small></b></div></div>}{prediction?.explanation && <div className="explanation-panel"><div className="explanation-heading"><span>Model explanation</span><small>{prediction.explanation.method}</small></div><div className="explanation-list">{prediction.explanation.contributions.map((item) => <span key={item.feature}><b>{item.feature}</b><em className={item.contribution >= 0 ? 'up' : 'down'}>{item.contribution >= 0 ? '+' : ''}{item.contribution}</em></span>)}</div><p>Unit: {prediction.explanation.unit}. Base {prediction.explanation.base_value} → output {prediction.explanation.output_value}. {prediction.explanation.faithfulness && <>Faithfulness check: deletion ratio {prediction.explanation.faithfulness.deletion_ratio ?? '—'}, ablation correlation {prediction.explanation.faithfulness.ablation_correlation ?? '—'} ({prediction.explanation.faithfulness.assessment.replaceAll('_', ' ')}). </>}{prediction.explanation.sampling.max_std_error !== undefined && <>Sampling error ≤ {prediction.explanation.sampling.max_std_error}. </>}{prediction.explanation.limitation}</p>{prediction.derived_features && <p>Derived from history: {Object.entries(prediction.derived_features).map(([key, value]) => `${key.replace('behavior_', '').replace('relationship_', '').replace('graph_', 'g_')}=${value}`).join(' · ')}</p>}<div className="audit-actions"><button onClick={() => recordAudit('confirmed_fraud')}>Record confirmed fraud</button><button onClick={() => recordAudit('marked_legitimate')}>Record legitimate</button></div></div>}</div></section></>}

      <section className="simulation-section">
        <div className="simulation-heading"><div><span className="section-label"><Activity size={14} /> Live + reference simulations</span><h2>See the model in motion.</h2><p>Explore the original FraudShield scenarios while your uploaded model stays separate, transparent, and ready for analysis.</p></div><button className="live-toggle" onClick={() => setLiveRunning((value) => !value)}>{liveRunning ? <Pause size={13} /> : <Play size={13} />} {liveRunning ? 'Pause live feed' : 'Resume live feed'}</button></div>
        <div className="simulation-layout"><div className="scenario-list">{simulations.map((scenario) => <button key={scenario.id} className={`scenario-card ${selectedSimulation?.id === scenario.id ? 'selected' : ''}`} onClick={() => setSelectedSimulation(scenario)}><span className={`scenario-dot scenario-${scenario.kind}`} /><div><strong>{scenario.label}</strong><small>{scenario.description}</small></div><ArrowRight size={14} /></button>)}</div><div className="simulation-focus"><div className="focus-top"><span className="focus-kicker">REFERENCE DECISION</span><span className={`decision-chip chip-${selectedSimulation?.kind || 'normal'}`}>{selectedSimulation?.kind === 'suspicious' ? 'Hold' : selectedSimulation?.kind === 'borderline' ? 'Step-up' : 'Approve'}</span></div><h3>{selectedSimulation?.label || 'Loading simulation'}</h3><p>{selectedSimulation?.description || 'Choose a reference event to inspect its context.'}</p><div className="focus-facts"><span><b>Amount</b>{selectedSimulation?.transaction?.amount ? `$${selectedSimulation.transaction.amount}` : '—'}</span><span><b>Location</b>{selectedSimulation?.transaction?.location || '—'}</span><span><b>Velocity</b>{selectedSimulation?.transaction?.velocity_10m ?? '—'} / 10m</span></div></div><div className="live-rail"><div className="rail-heading"><span>LIVE EVENT RAIL</span><i className={liveRunning ? 'rail-on' : ''} /> {liveRunning ? 'streaming' : 'paused'}</div>{liveEvents.length === 0 ? <div className="rail-empty">Waiting for events…</div> : liveEvents.map((event) => <div className="rail-event" key={`${event.id}-${event.stream_index}`}><span className={`scenario-dot scenario-${event.kind}`} /><div><strong>{event.label}</strong><small>{event.transaction?.location} · ${event.transaction?.amount}</small></div><b>{event.kind === 'suspicious' ? 'Hold' : event.kind === 'borderline' ? 'Step-up' : 'Approve'}</b></div>)}</div></div>
      </section>
      {(notice || error) && <div className={`toast ${error ? 'toast-error' : ''}`}><span>{error || notice}</span><button onClick={() => { setError(''); setNotice(''); }}><X size={14} /></button></div>}
      <section className="micro-footer"><span>FraudShield Z · transparent ML workflow</span><span>Simple by design · explainable by default</span></section>
    </main>
  </div>;
}

export default App;
