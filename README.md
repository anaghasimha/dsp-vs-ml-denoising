# Learning vs Model-Based Denoising under Noise-Level Mismatch

This repository studies how training data size and noise-level mismatch affect learned denoisers compared to classical DSP baselines.

## Motivation

Deep learning has become the dominant approach for signal denoising, often outperforming classical methods under matched training and test conditions. However, real-world signals frequently violate training assumptions, particularly with respect to noise level.

This project investigates a simple but underexplored question:

**Does increasing training data improve robustness to noise-level mismatch, or does it merely sharpen noise-specific learned priors?**

## Methods

We compare three classes of denoisers:

- **CNN denoiser**  
  A small convolutional neural network trained end-to-end using mean squared error.

- **Windowed linear regression**  
  A learned linear filter trained via least-squares over a fixed temporal window.

- **Classical DSP baselines**
  - Wiener filtering
  - Moving average filtering

All methods are evaluated on synthetic 1D signals with controlled additive noise.

## Experimental Setup

- Signal duration: 10 seconds  
- Sampling rate: 100 Hz  
- Train SNRs: −5, 0, 5, 10, 15, 20 dB  
- Test SNRs: −5, 0, 5, 10, 15, 20 dB  
- Training sizes: 50, 100, 200, 500, 1000, 2000  
- Test size: 200 signals per condition  

CNN and windowed linear models are trained separately for each training SNR and training size.
Classical DSP methods do not use training data.

Metrics:
- Mean Squared Error (MSE)
- SNR improvement relative to the noisy input

## Key Findings

1. **Learning curves behave as expected under sufficient training**
   - Increasing training size improves CNN and windowed linear denoisers under matched train/test SNR.
   - Undertraining artifacts observed at small epoch counts disappear with sufficient optimization.

2. **More data does not improve robustness to noise-level mismatch**
   - CNN denoisers trained at a fixed SNR degrade sharply when evaluated at mismatched noise levels.
   - This mismatch persists even at large training sizes and longer training.

3. **Metric mismatch reveals structural failures**
   - CNN MSE improves with training size across most conditions.
   - However, SNR improvement can collapse under mismatch, indicating oversmoothing and loss of signal structure.

4. **Classical DSP baselines remain stable**
   - Wiener filtering performance is invariant to training size and remains competitive under severe mismatch.
   - In some mismatched regimes, Wiener filtering outperforms learned denoisers.

5. **Model capacity influences failure mode**
   - Windowed linear models improve modestly and fail more uniformly.
   - CNNs achieve higher peak performance but exhibit sharper failure under mismatch.


## Visualizations

The repository produces:

- Heatmaps of MSE and SNR improvement across train/test SNRs
- Training-size curves showing performance vs dataset size
- Mismatch penalty heatmaps relative to matched training

These plots highlight both learning gains and robustness failures.


## Running the experiments

1. Install dependencies: pip install requirements.txt
2. Run the experiment script: run_sweep.py
3. Results are saved to: results.csv
4. Generate plots: plots.py


## Limitations

- Experiments are conducted on synthetic 1D signals.
- CNN architecture is intentionally small and not tuned for peak performance.
- Models are trained independently per noise level; multi-SNR or noise-conditioned training is not explored here.
- Results focus on MSE and SNR; perceptual metrics are not considered.

## Conclusion

These results suggest that increasing training data alone is insufficient to achieve robustness in learned denoisers. Without explicit modeling of uncertainty or noise conditions, learned priors can become overconfident and fail under distribution shift.

This highlights a fundamental trade-off between learned expressiveness and model-based robustness, motivating hybrid DSP–ML approaches and uncertainty-aware training.
