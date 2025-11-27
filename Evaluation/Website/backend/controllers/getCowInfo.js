const driveService = require("../models/driveServices.js");
const { generateCowGraph } = require("../utils/graphService");
const path = require("path");
const onnxPredictModel = require("../utils/onnxPredictModel.js");

exports.getCowInfo = async (req, res) => {
    try {
        const cowId = req.params.cowId;
        const data = await driveService.getCowDELBundle(cowId);

        const BCS = await onnxPredictModel.predict(data.imagePath);

        const estado = semaforo(BCS, data.DEL);

        //  Construir objeto para el EXE (lo que antes hacía dataset.json)
        const cowData = {
            img: data.imagePath,
            ID: data.cowId,
            DEL: data.DEL,
            BCS: BCS,
            Semaforo: estado
        };

        //  Generate graph with the cowData object
        const graphDataUri = await generateCowGraph(cowData);
        console.log("graphImg length:", graphDataUri.length);

        res.render("cowInfo", {
            cowID: data.cowId,
            bcs: BCS,
            diasLeche: data.DEL,
            estado,
            status: estado.toLowerCase(),
            graphImg: graphDataUri
        });

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: "Failed to generate graph" });
    }
};



function discretizeValue(x) {
    return Math.round(x * 100) / 100;   // same behavior as Python discretize_value
}

function normalRange(DEL) {
    if (DEL >= 0 && DEL <= 288) {
        const Max = discretizeValue(
            -1e-8 * DEL ** 3 +
            3e-5 * DEL ** 2 -
            0.0079 * DEL +
            3.2665
        );
        const Min = Max - 0.5;
        return { Max, Min };
    } 
    else if (DEL > 288 && DEL <= 500) {
        return { Max: 3.25, Min: 2.25 };
    }
    else {
        throw new Error("DEL fuera de rango (0–500)");
    }
}

function semaforo(BCS, DEL) {
    const { Max, Min } = normalRange(DEL);

    if (BCS >= Min && BCS <= Max) {
        return "Verde";
    } 
    else if (BCS >= Min - 0.25 && BCS <= Max + 0.25) {
        return "Amarillo";
    } 
    else {
        return "Rojo";
    }
}
