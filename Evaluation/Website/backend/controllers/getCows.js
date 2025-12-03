const driveService = require('../models/driveServices.js');
const path = require('path');
const { showErrorModal } = require("../utils/modalHelper");

exports.getCowData = async (req, res) => {
    try {
        let files;
        try {
            files = await driveService.getIdImag();
        } catch (err) {
            return showErrorModal(res, "cows", {
                errorType: "Error en el drive",
                errorMessage: "No se pudo acceder a los datos en Google Drive.",
                errorDetail: err.message,
                redirectUrl: "/",
                actionLabel: "Reintentar"
            });
        }

        if (!files || files.length === 0) {
            return showErrorModal(res, "cows", {
                errorType: "sin_archivos",
                errorMessage: "No se encontraron imágenes de vacas en Google Drive.",
                errorDetail: "La carpeta está vacía o los archivos no tienen formato válido.",
                redirectUrl: "/",
                actionLabel: "Actualizar"
            });
        }

        const cowFiles = files.filter(f =>
            f.cowId &&
            /^[0-9]{4}$/.test(f.cowId) &&
            f.mimeType?.startsWith("image/")
        );

        console.log("CowFiles filtrados:", cowFiles);

        if (cowFiles.length === 0) {
            return showErrorModal(res, "cows", {
                errorType: "sin_ids_validos",
                errorMessage: "No se encontraron IDs de vacas válidos.",
                errorDetail: "Aunque existen imágenes, ninguna contiene un ID válido asociado.",
                redirectUrl: "/",
                actionLabel: "Volver a intentar"
            });
        }
        
        const cowIds = [
            ...new Set(
                cowFiles
                    .map(f => f.cowId)
                    .filter(id => id !== null && id !== undefined)
            )
        ].sort();

        if (cowIds.length === 0) {
            return showErrorModal(res, "cows", {
                errorType: "ids_nulos",
                errorMessage: "Los archivos encontrados no tienen IDs válidos.",
                errorDetail: "Es posible que el naming de las imágenes no siga el formato esperado.",
                redirectUrl: "/",
                actionLabel: "Revisar Drive"
            });
        }

        const cows = cowIds.map(id => ({ IDCow: id }));

        return res.render("cows", {
            cows,
            filesFound: cowFiles.length,
            showError: false,
            showSuccess: false
        });

    } catch (error) {
        console.error("Error inesperado en getCowData:", error);
        return showErrorModal(res, "cows", {
            errorType: "error_interno",
            errorMessage: "Hubo un error inesperado cargando las vacas.",
            errorDetail: error.message,
            redirectUrl: "/",
            actionLabel: "Volver"
        });
    }
};