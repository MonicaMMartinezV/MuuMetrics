const { spawn } = require("child_process");
const path = require("path");

exports.getCowInfo = (req, res) => {
    const cowId = req.params.id;

    const scriptPath = path.join(__dirname, "..", "..", "python", "generateGraph.py");
    const jsonPath   = path.join(__dirname, "..", "..", "dataset.json");
    const graphPath  = path.join(__dirname, "..", "..", "frontend", "public", "images", `graph${cowId}.png`);

    console.log("⚡ Generando gráfica silenciosa…");
    console.log("Script:", scriptPath);
    console.log("JSON:  ", jsonPath);
    console.log("Output:", graphPath);

    // pythonw = silencioso SIEMPRE
    const pythonExec = path.join(__dirname, "..", "..", "venv", "Scripts", "pythonw.exe");

    try {
        const child = spawn(
            pythonExec,
            [scriptPath, jsonPath, cowId, graphPath],
            {
                windowsHide: true,
                shell: true,
                detached: true,
                stdio: "ignore"   // 🔥 NO logs → NO CMD → NO errores visuales
            }
        );

        child.unref(); // 🔥 lo suelta de PM2

    } catch (err) {
        console.error("❌ ERROR ejecutando pythonw:", err);
    }

    res.render("cowInfo", {
        cowID: cowId,
        bcs: 2.75,
        diasLeche: 143,
        estado: "Temporal",
        status: "rojo",
        graphImg: `/images/graph${cowId}.png`
    });
};