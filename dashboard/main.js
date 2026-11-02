const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

let mainWindow;
let pythonProcess;

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    },
    autoHideMenuBar: true,
    title: 'SiteSafe AI Dashboard'
  });

  mainWindow.loadURL(`http://localhost:${port}`);

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

function startPythonServer() {
  const scriptPath = path.join(__dirname, '..', 'ws_server.py');
  const pythonExecutable = path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe');
  
  console.log(`Starting Python server: ${pythonExecutable} ${scriptPath}`);
  
  pythonProcess = spawn(pythonExecutable, [scriptPath], {
    cwd: path.join(__dirname, '..'),
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });
}

function startStaticServer(callback) {
  const outDir = path.join(__dirname, 'out');
  
  const server = http.createServer((req, res) => {
    let filePath = path.join(outDir, req.url === '/' ? 'index.html' : req.url);
    if (!fs.existsSync(filePath)) {
      filePath = path.join(outDir, req.url + '.html');
    }
    
    fs.readFile(filePath, (err, content) => {
      if (err) {
        res.writeHead(404);
        res.end('File not found');
        return;
      }
      let contentType = 'text/html';
      const ext = path.extname(filePath);
      if (ext === '.js') contentType = 'text/javascript';
      if (ext === '.css') contentType = 'text/css';
      if (ext === '.json') contentType = 'application/json';
      if (ext === '.png') contentType = 'image/png';
      if (ext === '.jpg') contentType = 'image/jpg';
      
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content, 'utf-8');
    });
  });

  server.listen(0, '127.0.0.1', () => {
    callback(server.address().port);
  });
}

app.on('ready', () => {
  startStaticServer((port) => {
    startPythonServer();
    createWindow(port);
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill('SIGINT');
  }
});
