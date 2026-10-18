import fs from 'node:fs';
import path from 'node:path';
import { exportModelArtifact, hasTrainedModel, importModelArtifact } from './ml.js';

const dataDir = process.env.FRAUDSHIELD_DATA_DIR || path.resolve(process.cwd(), 'data');
const modelPath = process.env.FRAUDSHIELD_MODEL_PATH || (process.env.NODE_ENV === 'production' ? path.join(dataDir, 'active-model.json') : null);

export function restoreActiveModel() {
  if (!modelPath || !fs.existsSync(modelPath)) return false;
  const stat = fs.statSync(modelPath);
  if (!stat.isFile() || stat.size > 50 * 1024 * 1024) throw new Error('Stored model artifact is not a regular file or exceeds the 50 MB safety limit.');
  fs.chmodSync(modelPath, 0o600);
  const artifact = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
  importModelArtifact(artifact);
  return true;
}

export function persistActiveModel() {
  if (!modelPath || !hasTrainedModel()) return false;
  const dir = path.dirname(modelPath);
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  fs.chmodSync(dir, 0o700);
  const tempPath = `${modelPath}.${process.pid}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(exportModelArtifact()), { mode: 0o600, flag: 'w' });
  fs.renameSync(tempPath, modelPath);
  fs.chmodSync(modelPath, 0o600);
  return true;
}
