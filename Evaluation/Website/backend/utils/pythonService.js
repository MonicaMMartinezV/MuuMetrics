// services/pythonService.js
const { spawn } = require("child_process");
const { execFile } = require("child_process");
const path = require("path");

exports.runCowPythonScript = (cowId) => {
    const scriptPath = path.join(__dirname, "..", "python", "generateGraph.py");
    const jsonPath   = path.join(__dirname, "..", "dataset.json");
    const graphPath  = path.join(__dirname, "..", "frontend", "public", "images", `graph${cowId}.png`);

    const pythonExec = path.join(__dirname, "..", "venv", "Scripts", "pythonw.exe");

    console.log("⚡ Ejecutando script Python en background…"); 

    try {
        const child = spawn(
            pythonExec,
            [scriptPath, jsonPath, cowId, graphPath],
            {
                windowsHide: true,
                shell: true,
                detached: true,
                stdio: "ignore"
            }
        );

        child.unref(); // Release Python so it runs in background

    } catch (err) {
        console.error("❌ ERROR ejecutando pythonw:", err);
    }
};


function runProgram(datasetPath, cowId, outputPath) {
    return new Promise((resolve, reject) => {
        const exePath = path.join(__dirname, "..", "..", "generateGraph.exe");

        execFile(
            exePath,
            [datasetPath, cowId, outputPath],  // <-- pass the 3 arguments
            { cwd: path.dirname(exePath) },
            (error, stdout, stderr) => {
                if (error) {
                    console.error("EXE error:", stderr || error);
                    return reject(error);
                }
                resolve(stdout);
            }
        );
        console.log("✔ EXE process started");
    });
}


module.exports = {
    runProgram
};