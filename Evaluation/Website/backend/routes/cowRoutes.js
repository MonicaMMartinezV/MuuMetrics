const express = require('express');
const router = express.Router();
const controller = require('../controllers/controller');
//Hay que cambiar la ruta al controller correcto

router.get('/:id', controller.getCowInfo);
router.get('/', controller.refreshCowList);

module.exports = router; 