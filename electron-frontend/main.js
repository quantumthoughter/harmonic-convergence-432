const { app, BrowserWindow, dialog, ipcMain, session } = require('electron');
const path = require('path');
const os = require('os');
const http = require('http');
const fs = require('fs');

let mainWindow;
const API_URL = 'http://127.0.0.1:8765';

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#080816',
    show: false,
  });
  mainWindow.loadFile('index.html');
  mainWindow.once('ready-to-show', () => mainWindow.show());
}

app.whenReady().then(() => {
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    callback(['media','microphone','audioCapture'].includes(permission));
  });
  createWindow();
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });

ipcMain.handle('api:get', async (event, url) => await apiGet(url));
ipcMain.handle('api:post', async (event, url, body) => await apiPost(url, body));
ipcMain.handle('api:downloadBuffer', async (event, url) => await apiGetBuffer(url));
ipcMain.handle('dialog:openFile', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Audio', extensions: ['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac'] }],
  });
  return result.canceled ? null : result.filePaths[0];
});
ipcMain.handle('dialog:saveFile', async (event, defaultName) => {
  const ext = defaultName.split('.').pop();
  const isMp3 = ext === 'mp3';
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: path.join(os.homedir(), 'Desktop', '432_healed', defaultName),
    filters: isMp3 
      ? [{ name: 'MP3', extensions: ['mp3'] }]
      : [{ name: 'WAV', extensions: ['wav'] }],
  });
  return result.canceled ? null : result.filePath;
});
ipcMain.handle('dialog:selectFolder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, { properties: ['openDirectory'] });
  return result.canceled ? null : result.filePaths[0];
});
ipcMain.handle('fs:writeFile', async (event, filePath, data) => {
  fs.writeFileSync(filePath, Buffer.from(data));
  return true;
});
ipcMain.handle('fs:listAudio', async (event, folderPath) => {
  const exts = new Set(['.mp3','.wav','.flac','.ogg','.m4a','.aac']);
  try {
    return fs.readdirSync(folderPath).filter(f => exts.has(path.extname(f).toLowerCase())).map(f => path.join(folderPath, f));
  } catch { return []; }
});

async function apiGet(url) {
  return new Promise((resolve, reject) => {
    http.get(`${API_URL}${url}`, (res) => { let d = ''; res.on('data', c => d += c); res.on('end', () => { try { resolve(JSON.parse(d)); } catch { reject(new Error(d)); } }); }).on('error', reject);
  });
}
async function apiGetBuffer(url) {
  return new Promise((resolve, reject) => {
    http.get(`${API_URL}${url}`, (res) => { const c = []; res.on('data', d => c.push(d)); res.on('end', () => resolve(Buffer.concat(c))); }).on('error', reject);
  });
}
async function apiPost(url, body) {
  return new Promise((resolve, reject) => {
    const p = JSON.stringify(body);
    const req = http.request(`${API_URL}${url}`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(p) } }, (res) => { let d = ''; res.on('data', c => d += c); res.on('end', () => { try { resolve(JSON.parse(d)); } catch { reject(new Error(d)); } }); });
    req.on('error', reject); req.write(p); req.end();
  });
}
