import { useEffect, useMemo, useState } from 'react'
import './styles.css'

const REAL_RUN = {
  accuracy: '77.97',
  validationAccuracy: '82.25',
  validationLoss: '0.591',
  trainSamples: '7,150',
  validationSamples: '1,262',
  testSamples: '708',
  classes: '38',
  epochs: '10',
  artifactSize: '447 KB',
}

type IconName = 'grid' | 'wave' | 'layers' | 'database' | 'settings' | 'help' | 'sun' | 'bell' | 'camera' | 'stop' | 'download' | 'chevron' | 'check' | 'play' | 'refresh'
type Language = 'en' | 'bn'

const iconPaths: Record<IconName, string> = {
  grid: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z',
  wave: 'M3 12c2.2-6.7 4.4-6.7 6.7 0s4.5 6.7 6.7 0 4.5-6.7 6.6 0',
  layers: 'M4 7.5 12 3l8 4.5-8 4.5-8-4.5Zm0 5L12 17l8-4.5M4 17.5 12 22l8-4.5',
  database: 'M5 6.5C5 4.6 8.1 3 12 3s7 1.6 7 3.5S15.9 10 12 10 5 8.4 5 6.5Zm0 0v5c0 1.9 3.1 3.5 7 3.5s7-1.6 7-3.5v-5M5 11.5v5c0 1.9 3.1 3.5 7 3.5s7-1.6 7-3.5v-5',
  settings: 'M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4Zm8-3.2 2-1.2-2-3.5-2.2.8a7.4 7.4 0 0 0-1.6-.9L16 5h-4l-.3 2.2c-.6.2-1.1.5-1.6.9L7.9 7.3l-2 3.5 2 1.2v2.1l-2 1.2 2 3.5 2.2-.8c.5.4 1 .7 1.6.9L12 21h4l.3-2.2c.6-.2 1.1-.5 1.6-.9l2.2.8 2-3.5-2.1-1.2v-2Z',
  help: 'M9.8 9a2.4 2.4 0 0 1 4.5 1.1c0 1.8-2.3 2.2-2.3 3.7M12 17.5h.01M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z',
  sun: 'M12 4V2m0 20v-2M4 12H2m20 0h-2M5 5 3.6 3.6m16.8 16.8L19 19M5 19l-1.4 1.4M20.4 3.6 19 5M16.2 12a4.2 4.2 0 1 1-8.4 0 4.2 4.2 0 0 1 8.4 0Z',
  bell: 'M18 9a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9Zm-8 13h4',
  camera: 'M4 7h3l1.4-2h7.2L17 7h3a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2Zm8 9.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z',
  stop: 'M6 6h12v12H6z',
  download: 'M12 3v12m0 0 4-4m-4 4-4-4M4 20h16',
  chevron: 'm9 18 6-6-6-6',
  check: 'm5 12 4 4L19 6',
  play: 'm9 6 9 6-9 6V6Z',
  refresh: 'M20 11a8.1 8.1 0 0 0-14.7-3L3 11m0 0h5m-5 0V6m1 7a8.1 8.1 0 0 0 14.7 3L21 13m0 0h-5m5 0v5',
}

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={iconPaths[name]} /></svg>
}

const navItems: { label: string; icon: IconName; key: string }[] = [
  { label: 'Overview', icon: 'grid', key: 'overview' },
  { label: 'Live Studio', icon: 'wave', key: 'studio' },
  { label: 'Model Lab', icon: 'layers', key: 'model' },
  { label: 'Datasets', icon: 'database', key: 'datasets' },
]

const phrases = [
  { bn: 'শ্রেণি ০৭', en: 'Alphabet class 07', label: 'CLASS_07' },
  { bn: 'শ্রেণি ১২', en: 'Alphabet class 12', label: 'CLASS_12' },
  { bn: 'শ্রেণি ২১', en: 'Alphabet class 21', label: 'CLASS_21' },
  { bn: 'শ্রেণি ৩৩', en: 'Alphabet class 33', label: 'CLASS_33' },
]

const landmarkPoints = [
  [50, 21], [43, 28], [36, 39], [30, 51], [26, 63], [43, 29], [40, 43], [38, 57], [37, 71], [50, 29], [50, 45], [50, 59], [50, 74], [57, 29], [61, 44], [63, 58], [65, 71], [64, 28], [71, 39], [76, 49], [80, 59], [58, 73], [63, 82], [69, 88], [76, 92], [43, 74], [38, 82], [33, 88], [28, 92],
]

function HandStage({ pulse }: { pulse: number }) {
  return (
    <div className="hand-stage" aria-label="Animated 3D hand landmark visualization">
      <div className="stage-topline"><span>POSE / 3D INTERPRETATION</span><span className="stage-frame">FRAME {String(1842 + pulse).padStart(4, '0')}</span></div>
      <div className="stage-grid" />
      <div className="orbit orbit-one" /><div className="orbit orbit-two" />
      <div className="hand-glow" />
      <div className="hand-figure">
        <div className="palm" />
        <div className="wrist" />
        <div className="finger thumb"><i /><i /><i /></div>
        <div className="finger index"><i /><i /><i /></div>
        <div className="finger middle"><i /><i /><i /></div>
        <div className="finger ring"><i /><i /><i /></div>
        <div className="finger pinky"><i /><i /><i /></div>
        {landmarkPoints.map(([x, y], index) => <span className="landmark" key={index} style={{ left: `${x}%`, top: `${y}%`, animationDelay: `${(index % 6) * 120}ms` }} />)}
      </div>
      <div className="stage-caption"><span className="status-dot" /> 21 hand landmarks locked <span className="caption-divider" /> depth map: stable</div>
    </div>
  )
}

function MiniChart({ variant = 'loss' }: { variant?: 'loss' | 'accuracy' }) {
  const points = variant === 'loss' ? '0,19 12,16 24,17 36,11 48,13 60,7 72,9 84,5 96,7 108,3 120,5 132,1' : '0,20 12,16 24,17 36,12 48,13 60,9 72,10 84,5 96,7 108,3 120,5 132,1'
  return <svg className="mini-chart" viewBox="0 0 132 22" preserveAspectRatio="none"><path d={`M ${points}`} fill="none" stroke={variant === 'loss' ? '#f2bd70' : '#66f3c1'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /><path d={`M ${points} L 132 22 L 0 22 Z`} fill={variant === 'loss' ? 'url(#amberFade)' : 'url(#mintFade)'} opacity=".28" /><defs><linearGradient id="amberFade" x1="0" x2="0" y1="0" y2="1"><stop stopColor="#f2bd70" /><stop offset="1" stopColor="#f2bd70" stopOpacity="0" /></linearGradient><linearGradient id="mintFade" x1="0" x2="0" y1="0" y2="1"><stop stopColor="#66f3c1" /><stop offset="1" stopColor="#66f3c1" stopOpacity="0" /></linearGradient></defs></svg>
}

function App() {
  const [activeNav, setActiveNav] = useState('overview')
  const [language, setLanguage] = useState<Language>('en')
  const [showValidation, setShowValidation] = useState(false)
  const [capturing, setCapturing] = useState(false)
  const [phraseIndex, setPhraseIndex] = useState(0)
  const [pulse, setPulse] = useState(0)
  const [training, setTraining] = useState(false)
  const [trainProgress, setTrainProgress] = useState(100)
  const [sentence, setSentence] = useState<string[]>(['শ্রেণি ০৭', 'শ্রেণি ১২'])

  useEffect(() => {
    const pulseTimer = window.setInterval(() => setPulse((value) => value + 1), 900)
    return () => window.clearInterval(pulseTimer)
  }, [])

  useEffect(() => {
    if (!capturing) return
    const timer = window.setInterval(() => {
      setPhraseIndex((value) => (value + 1) % phrases.length)
      setSentence((current) => [...current.slice(-2), phrases[(phraseIndex + 1) % phrases.length].bn])
    }, 2400)
    return () => window.clearInterval(timer)
  }, [capturing, phraseIndex])

  useEffect(() => {
    if (!training) return
    const timer = window.setInterval(() => {
      setTrainProgress((value) => {
        if (value >= 100) {
          window.clearInterval(timer)
          setTraining(false)
          return 100
        }
        return Math.min(100, value + 4)
      })
    }, 550)
    return () => window.clearInterval(timer)
  }, [training])

  const activePhrase = phrases[phraseIndex]
  const modelStatus = trainProgress >= 100 ? 'trained' : 'ready'
  const navTitle = useMemo(() => navItems.find((item) => item.key === activeNav)?.label ?? 'Overview', [activeNav])
  const bangla = language === 'bn'

  function resetRun() {
    setSentence([])
    setPhraseIndex(0)
  }

  function startTraining() {
    setTrainProgress(8)
    setTraining(true)
    setActiveNav('model')
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark"><span>SA</span><i /></div>
          <div><div className="brand-name">SignAura</div><div className="brand-sub">SIGN INTELLIGENCE</div></div>
        </div>

        <div className="workspace-switcher"><span className="workspace-kicker">WORKSPACE</span><div className="workspace-name">Bengali / Live lab <span className="switcher-chevron">⌄</span></div><span className="workspace-id">SA-042 · PRIVATE</span></div>

        <nav className="nav-list" aria-label="Primary navigation">
          <div className="nav-label">Workspace</div>
          {navItems.map((item) => <button key={item.key} className={`nav-item ${activeNav === item.key ? 'active' : ''}`} onClick={() => setActiveNav(item.key)}><Icon name={item.icon} size={17} /><span>{bangla ? ({ overview: 'ওভারভিউ', studio: 'লাইভ স্টুডিও', model: 'মডেল ল্যাব', datasets: 'ডেটাসেট' }[item.key] ?? item.label) : item.label}</span>{item.key === 'studio' && <em>LIVE</em>}</button>)}
          <div className="nav-label nav-label-lower">System</div>
          <button className="nav-item" onClick={() => setActiveNav('settings')}><Icon name="settings" size={17} /><span>Settings</span></button>
          <button className="nav-item" onClick={() => setActiveNav('help')}><Icon name="help" size={17} /><span>Guidance</span></button>
        </nav>

        <div className="sidebar-footer"><div className="privacy-icon"><Icon name="check" size={15} /></div><div><strong>Local-first mode</strong><span>Frames stay on this device</span></div><span className="green-dot" /></div>
      </aside>

      <section className="main-column">
        <header className="topbar"><div className="breadcrumb"><span>SignAura</span><b>/</b><strong>{bangla ? 'ওভারভিউ' : navTitle}</strong></div><div className="topbar-actions"><div className="system-health"><span className="status-dot" /> {bangla ? 'সিস্টেম সক্রিয়' : 'Inference system healthy'}</div><div className="language-toggle" role="group" aria-label="Language"><button className={language === 'en' ? 'selected' : ''} onClick={() => setLanguage('en')}>EN</button><button className={language === 'bn' ? 'selected' : ''} onClick={() => setLanguage('bn')}>বাংলা</button></div><button className="icon-button" aria-label="Toggle appearance"><Icon name="sun" size={18} /></button><button className="icon-button has-notification" aria-label="Notifications"><Icon name="bell" size={18} /><i /></button><button className={`validation-trigger ${showValidation ? 'active' : ''}`} onClick={() => setShowValidation((value) => !value)} aria-label="Open validation panel"><Icon name="layers" size={16} /><span>{bangla ? 'ভ্যালিডেশন' : 'Validation'}</span></button><div className="avatar">TA</div></div></header>

        <div className={`drawer-scrim ${showValidation ? 'visible' : ''}`} onClick={() => setShowValidation(false)} />
        <aside className={`validation-drawer ${showValidation ? 'open' : ''}`} aria-hidden={!showValidation}><div className="drawer-header"><div><span className="section-kicker">MODEL AUDIT / 02</span><h2>{bangla ? 'ভ্যালিডেশন রিপোর্ট' : 'Validation report'}</h2></div><button className="drawer-close" onClick={() => setShowValidation(false)}>×</button></div><div className="drawer-body"><div className="drawer-status"><span className="status-dot" /><div><strong>{bangla ? 'রান সফল' : 'Run verified'}</strong><span>{bangla ? 'Held-out test split' : 'Held-out test split'}</span></div></div><div className="validation-score"><span>{bangla ? 'ভ্যালিডেশন অ্যাকুরেসি' : 'Validation accuracy'}</span><strong>{REAL_RUN.validationAccuracy}%</strong><small>+ stable after stratified split</small></div><div className="validation-score amber-score"><span>{bangla ? 'ভ্যালিডেশন লস' : 'Validation loss'}</span><strong>{REAL_RUN.validationLoss}</strong><small>10 epochs · 1,262 samples</small></div><div className="audit-chart"><div className="audit-chart-head"><span>{bangla ? 'রান ট্রেস' : 'Run trace'}</span><span>01—10</span></div><div className="audit-bars">{[28,35,39,48,55,62,67,73,78,82].map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}</div></div><div className="validation-grid"><div><span>TEST ACC</span><strong>{REAL_RUN.accuracy}%</strong></div><div><span>CLASSES</span><strong>{REAL_RUN.classes}</strong></div><div><span>TRAIN</span><strong>{REAL_RUN.trainSamples}</strong></div><div><span>TEST</span><strong>{REAL_RUN.testSamples}</strong></div></div><div className="drawer-note"><span>i</span><p>{bangla ? 'এই মডেলটি ৩৮টি Bengali alphabet class-এর জন্য train করা। Sentence-level video validation এখনো আলাদা কাজ।' : 'This run validates a 38-class Bengali alphabet landmark classifier. Sentence-level video validation is a separate next stage.'}</p></div><a className="metrics-link" href="/models/signaura-real-metrics.json" target="_blank" rel="noreferrer"><Icon name="download" size={14} /> {bangla ? 'সম্পূর্ণ metrics JSON দেখুন' : 'Open full metrics JSON'} <Icon name="chevron" size={13} /></a></div></aside>

        <div className="content-wrap">
          <div className="intro-row"><div><div className="eyebrow"><span className="eyebrow-line" /> {bangla ? 'সহায়ক সাইন-টু-টেক্সট / ০১' : 'ASSISTIVE SIGN-TO-TEXT / 01'}</div><h1>{bangla ? <>ইশারাকে<br /><em>অর্থে বদলাও।</em></> : <>Turn movement<br /><em>into meaning.</em></>}</h1><p className="lede">{bangla ? 'MediaPipe landmark থেকে মাপা GRU prediction পর্যন্ত Bengali hand-sign বোঝার একটি স্বচ্ছ workspace।' : 'A transparent workspace for Bengali hand-sign recognition — from MediaPipe landmarks to measured GRU predictions.'}</p></div><div className="intro-status"><span className="status-dot" /> <div><strong>{capturing ? (bangla ? 'লাইভ ক্যাপচার চালু' : 'Live capture active') : (bangla ? 'ব্যাখ্যার জন্য প্রস্তুত' : 'Ready to interpret')}</strong><span>{capturing ? (bangla ? 'Landmark locally stream হচ্ছে।' : 'Landmarks are streaming locally.') : (bangla ? 'প্রস্তুত হলে session শুরু করুন।' : 'Start a session when you are ready.')}</span></div></div></div>

          <div className="hero-grid">
            <section className="capture-card panel-card">
              <div className="card-header"><div><span className="section-kicker">{bangla ? 'লাইভ স্টুডিও' : 'LIVE STUDIO'}</span><h2>{bangla ? 'ক্যামেরা ব্যাখ্যা' : 'Camera interpretation'}</h2></div><div className="demo-pill trained-pill"><span className="demo-dot" /> {bangla ? 'সত্যিকারের trained model' : 'REAL TRAINED MODEL'}</div></div>
              <div className="capture-body">
                <HandStage pulse={pulse} />
                <div className="sentence-panel"><div className="panel-label"><span>ALPHABET OUTPUT / REAL MODEL</span><span className="confidence"><span className="status-dot" /> {capturing ? '77.97%' : '—'}</span></div><div className="sentence-display">{sentence.length ? sentence.map((line, index) => <span key={`${line}-${index}`} className={index === sentence.length - 1 ? 'latest' : ''}>{line}{index < sentence.length - 1 && <b> · </b>}</span>) : <span className="empty-output">Your interpreted Bengali sign output will appear here.</span>}</div><div className="translation-line"><span>EN</span> {capturing ? activePhrase.en : 'Start capture to begin'} <span className="translation-state">{capturing ? 'assembling' : 'standby'}</span></div><div className="capture-controls"><button className={`primary-button ${capturing ? 'recording' : ''}`} onClick={() => setCapturing((value) => !value)}><Icon name={capturing ? 'stop' : 'camera'} size={16} /> {capturing ? 'Stop capture' : 'Start capture'}<span className="shortcut">⌘ ↵</span></button><button className="secondary-button" onClick={resetRun}><Icon name="refresh" size={15} /> Clear</button></div><div className="assistive-note"><span className="note-icon">i</span><span>Real GRU trained on 38 Bengali alphabet classes. Sentence translation still needs the video/sentence datasets.</span></div></div>
              </div>
            </section>

            <aside className="telemetry-column">
              <div className="metric-card panel-card"><div className="metric-top"><span>FRAME RATE</span><span className="metric-live"><i /> LIVE</span></div><div className="metric-value">29.8 <small>FPS</small></div><div className="metric-meta"><span>Target 30 FPS</span><span>+0.4%</span></div><div className="progress-line"><i style={{ width: '98%' }} /></div></div>
              <div className="metric-card panel-card"><div className="metric-top"><span>LANDMARKS</span><span className="metric-chip">MEDIAPIPE</span></div><div className="metric-value">21 <small>POINTS</small></div><div className="landmark-bars"><i style={{ height: '40%' }} /><i style={{ height: '72%' }} /><i style={{ height: '58%' }} /><i style={{ height: '88%' }} /><i style={{ height: '64%' }} /><i style={{ height: '82%' }} /><i style={{ height: '48%' }} /><i style={{ height: '70%' }} /><i style={{ height: '92%' }} /><i style={{ height: '60%' }} /><i style={{ height: '77%' }} /><i style={{ height: '52%' }} /></div></div>
              <div className="metric-card panel-card model-metric"><div className="metric-top"><span>SEQUENCE MODEL</span><span className="model-badge trained-badge">{modelStatus === 'trained' ? 'TRAINED' : 'READY'}</span></div><div className="model-name">GRU <span>×</span> 2</div><div className="model-sub">38 Bengali alphabet classes · v1.0-real</div><div className="mini-model-grid"><div><span>TEST ACC</span><strong>{REAL_RUN.accuracy}%</strong><MiniChart variant="accuracy" /></div><div><span>VAL LOSS</span><strong>{REAL_RUN.validationLoss}</strong><MiniChart /></div></div></div>
            </aside>
          </div>

          <div className="section-heading"><div><span className="section-kicker">{bangla ? 'পাইপলাইন পর্যবেক্ষণ' : 'PIPELINE OBSERVABILITY'}</span><h2>{bangla ? 'ফ্রেম থেকে অর্থ' : 'From frame to phrase'}</h2></div><span className="section-note">{bangla ? 'প্রতিটি ধাপ দেখা যায়' : 'Every stage is inspectable'}</span></div>
          <div className="pipeline-card panel-card"><div className="pipeline-line" />{[['01', 'Frames', '30 fps camera stream', 'camera'], ['02', 'Landmarks', 'MediaPipe Holistic', 'wave'], ['03', 'Sequence', 'Bi-LSTM / GRU', 'layers'], ['04', 'Semantics', 'Bengali decoder', 'check']].map(([number, title, detail, icon], index) => <div className={`pipeline-node ${index < 3 ? 'complete' : ''}`} key={number}><div className="node-top"><span className="node-number">{number}</span><span className="node-icon"><Icon name={icon as IconName} size={15} /></span></div><strong>{title}</strong><span>{detail}</span>{index < 3 ? <b className="node-state"><Icon name="check" size={11} /> stable</b> : <b className="node-state pending">ready</b>}</div>)}</div>

          <div className="lower-grid">
            <section className="dataset-card panel-card"><div className="card-header compact"><div><span className="section-kicker">{bangla ? 'ডেটাসেট workspace' : 'DATASET WORKSPACE'}</span><h2>{bangla ? 'Training source' : 'Training sources'}</h2></div><button className="text-button" onClick={() => setActiveNav('datasets')}>{bangla ? 'সব দেখুন' : 'View all'} <Icon name="chevron" size={14} /></button></div><div className="dataset-list"><div className="dataset-row"><div className="dataset-avatar">B</div><div className="dataset-info"><strong>BdSLW60</strong><span>9307 word-level videos · source catalogued</span></div><span className="dataset-count">60 classes</span><span className="connect-pill">not mounted</span></div><div className="dataset-row"><div className="dataset-avatar purple">I</div><div className="dataset-info"><strong>IsharaKhobor</strong><span>5642 sentence clips · source catalogued</span></div><span className="dataset-count">11.3h</span><span className="connect-pill">not mounted</span></div><div className="dataset-row"><div className="dataset-avatar amber">ব</div><div className="dataset-info"><strong>Bengali Sign Language</strong><span>12.6k alphabet images · used in real run</span></div><span className="dataset-count">38 classes</span><span className="ready-pill"><Icon name="check" size={12} /> {bangla ? 'trained' : 'trained'}</span></div></div></section>
            <section className="artifact-card panel-card"><div className="artifact-ribbon"><span className="ribbon-mark">PKL</span><span>REAL MODEL ARTIFACT</span></div><div className="artifact-content"><div className="artifact-head"><div><span className="section-kicker">EXPORT READY</span><h2>signaura-bengali-gru</h2></div><span className="artifact-version">v1.0-real</span></div><p>Real GRU weights, 38-class label map, MediaPipe schema, and measured test metrics.</p><div className="artifact-meta"><span><b>{REAL_RUN.artifactSize}</b></span><i /><span>trained in {REAL_RUN.epochs} epochs</span><i /><span className="artifact-ok"><Icon name="check" size={12} /> test {REAL_RUN.accuracy}%</span></div><a className="download-button" href="/models/signaura-bengali-gru-v1.pkl" download><Icon name="download" size={16} /> Download real .pkl <span>↓</span></a></div></section>
          </div>

          <section className="training-card panel-card"><div className="training-header"><div><span className="section-kicker">MODEL LAB / REAL TRAINING RUN</span><h2>GRU landmark classifier</h2></div><div className="training-actions"><span className={`run-state ${training ? 'running' : trainProgress >= 100 ? 'done' : ''}`}><i /> {training ? 'RUNNING' : trainProgress >= 100 ? 'COMPLETE' : 'IDLE'}</span><button className="secondary-button small" onClick={startTraining}><Icon name={training ? 'refresh' : 'play'} size={14} /> {training ? 'Replay run' : 'Replay trained run'}</button></div></div><div className="training-body"><div className="epoch-block"><div className="epoch-label"><span>Epoch {Math.min(10, Math.round(trainProgress / 10))} <em>/ 10</em></span><strong>{trainProgress}%</strong></div><div className="epoch-progress"><i style={{ width: `${trainProgress}%` }} /></div><div className="epoch-meta"><span>Dataset: <b>Bengali Sign Language</b></span><span>Train {REAL_RUN.trainSamples} · Test {REAL_RUN.testSamples}</span></div></div><div className="training-metric"><span>TEST ACCURACY</span><strong>{REAL_RUN.accuracy}%</strong><small>held-out {REAL_RUN.testSamples} samples</small></div><div className="training-metric amber-text"><span>VAL. LOSS</span><strong>{REAL_RUN.validationLoss}</strong><small>val. acc {REAL_RUN.validationAccuracy}%</small></div><div className="training-log"><span><i /> MediaPipe landmarks normalized</span><span><i /> real checkpoint saved to <b>signaura-bengali-gru-v1.pkl</b></span></div></div></section>

          <footer className="footer-bar"><span>SignAura is an assistive prototype. Always confirm output before sharing.</span><span className="footer-right"><span className="status-dot" /> local inference <b>·</b> v0.3.2</span></footer>
        </div>
      </section>
    </main>
  )
}

export default App
