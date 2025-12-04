const driveService = require("../models/driveServices.js");
const { generateCowGraph } = require("../utils/graphService");
const onnxPredictModel = require("../utils/onnxPredictModel.js");
const { showErrorModal, showSuccessModal} = require("../utils/modalHelper");

exports.getCowInfo = async (req, res) => {
    const cowId = req.params.cowId;
    try {
        const data = await driveService.getCowDELBundle(cowId);
        console.log("Datos obtenidos de Drive para la vaca:", data.cowId);
        if (!data.cowId) {
            return showErrorModal(res, "cows", {
                errorType: "error_drive",
                errorMessage: `No se pudieron cargar los datos de la vaca ${cowId}.`,
                errorDetail: "Faltan archivos o hubo un fallo en Drive.",
                redirectUrl: "/",
                actionLabel: "Volver"
            });
        }


        if (!data.imagePath) {
            return showErrorModal(res, "cows", {
                errorType: "Imagen no encontrada",
                errorMessage: `La imagen asociada a la vaca ${cowId} no existe en Drive.`,
                errorDetail: "No se encontró ninguna imagen asociada a la vaca",
                redirectUrl: "/",
                actionLabel: "Regresar"
            });
        }

        let BCS;
        try {
            BCS = await onnxPredictModel.predict(data.imagePath);
        } catch (err) {
            return showErrorModal(res, "cows", {
                errorType: "Fallo en el modelo",
                errorMessage: "El modelo no pudo procesar la imagen",
                errorDetail: err.message,
                redirectUrl: "/",
                actionLabel: "Regresar"
            });
        }

        const estado = semaforo(BCS, data.DEL);

        let graphDataUri;
        try {
            graphDataUri = await generateCowGraph({
                img: data.imagePath,
                ID: data.cowId,
                DEL: data.DEL,
                BCS,
                Semaforo: estado
            });

        } catch (err) {
            console.log("ERROR al generar gráfica:", err);
            return showErrorModal(res, "cows", {
                errorType: "grafica_fallo",
                errorMessage: "No fue posible generar la gráfica.",
                errorDetail: err.message,
                redirectUrl: "/cows",
                actionLabel: "Volver"
            });
        }
        console.log("Gráfica generada exitosamente para la vaca:", data.cowId);
        return res.render("cowInfo", {
            successMessage: "Datos del Drive cargados exitosamente.",
            errorMessage: null,
            errorDetail: null,
            errorAction: null, 
            cowID: data.cowId,
            bcs: BCS,
            diasLeche: data.DEL,
            estado,
            status: estado.toLowerCase(),
            graphImg: graphDataUri,
            showSuccess: true,  
        });


    } catch (err) {
        console.error("ERROR getCowInfo():", err);
        return showErrorModal(res, "cows", {
            errorType: "Error interno",
            errorMessage: "Ocurrió un error interno procesando la información.",
            errorDetail: err.message,
            redirectUrl: "/",
            actionLabel: "Volver al inicio"
        });
    }
};

function discretizeValue(x) {
    return Math.round(x * 100) / 100;   
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

exports.showCowLoader = (req, res) => {
    res.render("components/loader", {
        redirectUrl: `/cow/${req.params.cowId}/info`
    });
};
