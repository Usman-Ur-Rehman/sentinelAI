class AutoRetrainer:
  
    def retrain_and_swap(self, X_new):
        

        # atomic hot-swap — lock prevents inference using model mid-swap
        with self.swap_lock:
            self.live_model = new_model
            joblib.dump(new_model, self.model_path)              # persist to disk
            self.driftDetector.drift_detected = False            # reset drift flag
            self.driftDetector.drift_reason = None               # reset reason
            self.driftDetector.drifted_features = []             # reset drifted features list
            self.driftDetector.avg_confidence_score = 1.0        # reset to best confidence
            self.driftDetector.confidence_scores.clear()         # clear rolling confidence window
            for f in self.driftDetector.feature_windows:         # clear all 12 feature windows
                self.driftDetector.feature_windows[f].clear()

        print('[AutoRetrainer] Hot-swap complete. All drift state reset.')

    def monitor_loop(self, X_data):
        # runs in background thread — checks drift every 60 seconds
        while True:
            time.sleep(60)
            if self.driftDetector.drift_detected:  # if drift flagged — retrain
                self.retrain_and_swap(X_data)