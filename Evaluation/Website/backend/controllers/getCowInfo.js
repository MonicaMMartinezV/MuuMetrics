const pythonService = require("../services/pythonService");
const driveService = require("../models/driveServices.js");

exports.getCowInfo = (req, res) => {
    const cowId = req.params.id;

    driveService.downloadImageAndCsv(cowId);
    
    // 🟩 Run Python in background (clean)
    pythonService.runCowPythonScript(cowId);

    // 🟩 Render page normally
    res.render("cowInfo", {
        cowID: cowId,
        bcs: 2.75,
        diasLeche: 143,
        estado: "Temporal",
        status: "rojo",
        graphImg: "/images/graph${cowId}.png"
    });
};