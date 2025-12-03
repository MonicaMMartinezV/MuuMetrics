const driveService = require('../models/driveServices.js');
const path = require('path');
const { showErrorModal, showSuccessModal } = require("../utils/modalHelper");

exports.getCowData = async (req, res) => {
    try {
        let files;

        // --- ERROR AL CARGAR DRIVE ---
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
            f.cowId &&
            /^[0-9]{4}$/.test(f.cowId) &&
            f.mimeType?.startsWith("image/")
        );

        console.log("CowFiles filtrados:", cowFiles);

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

        // --- EXTRAER IDS ÚNICOS ---
        const cowIds = [
            ...new Set(cowFiles.map(f => f.cowId))
        ].sort();

        if (cowIds.length === 0) {
            return showErrorModal(res, "cows", {
                errorType: "ids_nulos",
                errorMessage: "Los archivos encontrados no tienen IDs válidos.",
                errorDetail: "El naming de las imágenes podría no seguir el formato esperado.",
                redirectUrl: "/",
                actionLabel: "Revisar Drive",
                cows: []
            });
        }

        const cows = cowIds.map(id => ({ IDCow: id }));

        // --- MODAL DE ÉXITO ---
        return showSuccessModal(res, "cows", {
            successMessage: "Datos cargados correctamente desde Drive.",
            redirectUrl: "/",
            actionLabel: "Continuar",
            cows,                    // 🔥 NECESARIO PARA QUE NO TRUENE
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
