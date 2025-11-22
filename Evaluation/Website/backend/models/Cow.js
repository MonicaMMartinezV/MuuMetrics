module.exports = (sequelize, DataTypes) => {
  const CowInfo = sequelize.define(
    'CowInfo',
    {
      IDCowInfo: {
        type: DataTypes.BIGINT,
        primaryKey: true,
        autoIncrement: true,
      },
      DEL: {
        type: DataTypes.BIGINT,
        allowNull: false,
        comment: 'Días en leche',
      },
      BCS: {
        type: DataTypes.DOUBLE,
        allowNull: false,
        comment: 'Body Condition Score',
      },
      DateMilking: {
        type: DataTypes.DATEONLY,
        allowNull: true,
        comment: 'Último día de ordeño',
      },
      IDCow: {
        type: DataTypes.BIGINT,
        allowNull: false,
        comment: 'Identificación del animal',
      },
    },
    {
      tableName: 'cowInfo',
      timestamps: false,
    }
  );

  return CowInfo;
};