exports.showErrorModal = function (res, view, data = {}) {
    return res.render(view, {
        showError: true,
        showSuccess: false,
        errorType: data.errorType || "generic",
        errorMessage: data.errorMessage || "Ocurrió un error.",
        errorDetail: data.errorDetail || null,
        errorAction: data.errorAction || null,

        // 🔥 IMPORTANTÍSIMO: evitar crash en cows.ejs
        cows: data.cows || [],
        filesFound: data.filesFound || 0,

        // Para que no reviente el modal si no hay success
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
