const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('api', {
  get: (url) => ipcRenderer.invoke('api:get', url),
  post: (url, body) => ipcRenderer.invoke('api:post', url, body),
  downloadBuffer: (url) => ipcRenderer.invoke('api:downloadBuffer', url),
  openFile: () => ipcRenderer.invoke('dialog:openFile'),
  saveFile: (name) => ipcRenderer.invoke('dialog:saveFile', name),
  selectFolder: () => ipcRenderer.invoke('dialog:selectFolder'),
  writeFile: (path, data) => ipcRenderer.invoke('fs:writeFile', path, data),
  listAudio: (path) => ipcRenderer.invoke('fs:listAudio', path),
  readFile: (path) => ipcRenderer.invoke('fs:readFile', path),
  getPathForFile: (file) => webUtils ? webUtils.getPathForFile(file) : (file?.path || null),
});
