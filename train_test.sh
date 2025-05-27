source ml_env/bin/activate

echo "Creating embeddings from training data ..."
python embeddings_creation_and_training_vggface2.py embed --csv data/data.csv --imgdir data/Images --split train --out embeddings_train.npz --batch 64 --num_workers 3
echo "Creating embeddings from training data completed."

echo "Training XgBoost Regressor ..."
python embeddings_creation_and_training_vggface2.py train_xgb --npz embeddings_train.npz --model_out xgb_bmi.json
echo "Training XgBoost Regressor completed."

echo "Evaluating XgBoost Regressor ..."
python embeddings_creation_and_training_vggface2.py evaluate_xgb --csv data/data.csv --imgdir data/Images --model xgb_bmi.json --out test_predictions.csv
echo "Evaluating Completed"