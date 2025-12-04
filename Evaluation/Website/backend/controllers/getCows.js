const driveService = require('../models/driveServices.js');
const path = require('path');
const { showErrorModal, showSuccessModal } = require("../utils/modalHelper");
const { getFiles} = require('../models/driveServices.js');

exports.getCowData = async (req, res) => {
    try {

        const files = await getFiles();

        // --- NO HAY ARCHIVOS ---
        if (!files || files.length === 0) {
            return showErrorModal(res, "cows", {
                errorType: "sin_archivos",
                errorMessage: "No se encontraron imágenes de vacas en Google Drive.",
                errorDetail: "La carpeta está vacía o los archivos no tienen formato válido.",
                redirectUrl: "/",
                actionLabel: "Actualizar",
                cows: []
            });
        }
        // --- FILTRAR SOLO IMÁGENES CON cowId ---
        const cowFiles = files.filter(f =>
            /^[0-9]{4}/.test(f.name) && f.mimeType.startsWith("image/")
        );

        if (cowFiles.length === 0) {
            return showErrorModal(res, "cows", {
                errorType: "sin_ids_validos",
                errorMessage: "No se encontraron IDs de vacas válidos.",
                errorDetail: "Las imágenes no contienen IDs válidos asociados.",
                redirectUrl: "/",
                actionLabel: "Volver a intentar",
                cows: []
            });
        }

        // Extract unique cow IDs
        let uniqueCowIds = [...new Set(cowFiles.map(f => f.name.substring(0, 4)))];

        if (uniqueCowIds.length === 0) {
            return showErrorModal(res, "cows", {
                errorType: "ids_nulos",
                errorMessage: "Los archivos encontrados no tienen IDs válidos.",
                errorDetail: "El naming de las imágenes podría no seguir el formato esperado.",
                redirectUrl: "/",
                actionLabel: "Revisar Drive",
                cows: []
            });
        }

        const cows = uniqueCowIds.map(id => ({ IDCow: id }));

        console.log("CowFiles filtrados:", cowFiles);
        return showSuccessModal(res, "cows", {
            successMessage: "Datos cargados correctamente desde Drive.",
            redirectUrl: "/",
            actionLabel: "Continuar",
            cows,
            filesFound: cowFiles.length
        });

    } catch (error) {
        console.error("Error inesperado en getCowData:", error);

        return showErrorModal(res, "cows", {
            errorType: "error_interno",
            errorMessage: "Hubo un error inesperado cargando las vacas.",
            errorDetail: error.message,
            redirectUrl: "/",
            actionLabel: "Volver",
            cows: []
        });
    }
};
