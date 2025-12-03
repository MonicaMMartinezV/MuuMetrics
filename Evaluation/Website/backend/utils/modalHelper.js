exports.showErrorModal = function (res, view, {
    errorType = "error",
    errorMessage = "Ha ocurrido un error",
    errorDetail = "",
    redirectUrl = "/",
    actionLabel = "Aceptar"
} = {}) {
    return res.render(view, {
        showError: true,
        showSuccess: false,
        errorType,
        errorMessage,
        errorDetail,
        errorAction: {
            label: actionLabel,
            url: redirectUrl
        }
    });
};

exports.showSuccessModal = function (res, view, {
    successMessage = "Operación exitosa",
    redirectUrl = "/",
    actionLabel = "Continuar"
} = {}) {
    return res.render(view, {
        showError: false,
        showSuccess: true,
        successMessage,
        successAction: {
            label: actionLabel,
            url: redirectUrl
        }
    });
};