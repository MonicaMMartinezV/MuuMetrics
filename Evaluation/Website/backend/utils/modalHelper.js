exports.showErrorModal = function (res, view, data = {}) {
    return res.render(view, {
        showError: true,
        showSuccess: false,

        errorType: data.errorType || "Error",
        errorMessage: data.errorMessage || "Ocurrió un error.",
        errorDetail: data.errorDetail || "",

        errorAction: {
            label: data.actionLabel || "Aceptar",
            url: data.redirectUrl || ""
        },

        cows: data.cows || [],
        filesFound: data.filesFound || 0,

        successMessage: null,
        successAction: null
    });
};

exports.showSuccessModal = function (res, view, data = {}) {
    return res.render(view, {
        showError: false,
        showSuccess: true,

        successMessage: data.successMessage || "Operación exitosa",
        successAction: {
            label: data.actionLabel || "Continuar",
            url: data.redirectUrl || "/"
        },

        // 🔥 IMPORTANTE — Estos deben pasarse SIEMPRE
        cows: data.cows || [],
        filesFound: data.filesFound || 0,

        // Para evitar reventar la vista
        errorType: null,
        errorMessage: null,
        errorDetail: null,
        errorAction: null
    });
};
