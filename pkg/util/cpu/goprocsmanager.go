// Package gomaxprocsmanager provides functionality to dynamically adjust GOMAXPROCS
// based on lock contention metrics within a Go application.
//
// It monitors the `/sync/mutex/wait/total:seconds` metric to determine the
// cumulative time goroutines spend blocked on mutexes or runtime-internal locks.
// Based on configurable thresholds, it adjusts GOMAXPROCS to optimize performance.
package cpu

import (
	"context"
	"fmt"
	"runtime"
	"runtime/metrics"
	"sync"
	"time"

	"github.com/pingcap/log"
	"go.uber.org/zap"
)

// Config holds the configuration parameters for the GOMAXPROCS manager.
type Config struct {
	// MonitorInterval defines how often the lock contention metrics are checked.
	MonitorInterval time.Duration

	// AdjustmentCooldown specifies the minimum duration between consecutive adjustments.
	AdjustmentCooldown time.Duration

	// HighLockWaitRateThreshold is the lock wait rate above which GOMAXPROCS will be decreased.
	HighLockWaitRateThreshold float64

	// LowLockWaitRateThreshold is the lock wait rate below which GOMAXPROCS will be increased.
	LowLockWaitRateThreshold float64

	// MinProcs defines the minimum allowed value for GOMAXPROCS.
	MinProcs int

	// MaxProcs defines the maximum allowed value for GOMAXPROCS.
	MaxProcs int
}

// Manager manages the dynamic adjustment of GOMAXPROCS based on lock contention.
type Manager struct {
	config         Config
	cancelFunc     context.CancelFunc
	wg             sync.WaitGroup
	mutex          sync.Mutex
	currentProcs   int
	prevData       MutexWaitData
	lastAdjustment time.Time
}

// MutexWaitData holds timestamped mutex wait total.
type MutexWaitData struct {
	timestamp time.Time
	total     float64
}

// NewManager creates a new Manager with the provided configuration.
// It returns an error if the configuration is invalid.
func NewManager(cfg Config) (*Manager, error) {
	// Validate configuration
	if cfg.MinProcs < 1 {
		return nil, fmt.Errorf("MinProcs must be at least 1")
	}
	if cfg.MaxProcs < cfg.MinProcs {
		return nil, fmt.Errorf("MaxProcs must be greater than or equal to MinProcs")
	}
	if cfg.MonitorInterval <= 0 {
		return nil, fmt.Errorf("MonitorInterval must be positive")
	}
	if cfg.AdjustmentCooldown < 0 {
		return nil, fmt.Errorf("AdjustmentCooldown cannot be negative")
	}

	manager := &Manager{
		config:         cfg,
		currentProcs:   runtime.GOMAXPROCS(0),                   // Get current GOMAXPROCS without setting
		lastAdjustment: time.Now().Add(-cfg.AdjustmentCooldown), // Initialize to allow immediate adjustment
	}

	return manager, nil
}

// Start begins the monitoring and adjustment process.
// It runs in the background and does not block.
func (m *Manager) Start() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if m.cancelFunc != nil {
		// Already started
		log.Info("GOMAXPROCS manager is already running")
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	m.cancelFunc = cancel
	m.wg.Add(1)
	go m.monitorAndAdjust(ctx)
	log.Info("GOMAXPROCS manager started")
}

// Stop terminates the monitoring and adjustment process gracefully.
// It waits for all goroutines to finish.
func (m *Manager) Stop() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if m.cancelFunc == nil {
		// Not started
		log.Info("GOMAXPROCS manager is not running")
		return
	}

	m.cancelFunc()
	m.cancelFunc = nil
	m.wg.Wait()
	log.Info("GOMAXPROCS manager stopped")
}

// monitorAndAdjust continuously monitors lock contention and adjusts GOMAXPROCS accordingly.
func (m *Manager) monitorAndAdjust(ctx context.Context) {
	defer m.wg.Done()

	// Initialize previous data
	total, err := m.getMutexWaitTotal()
	if err != nil {
		log.Error("Error initializing mutex wait total", zap.Error(err))
		total = 0
	}
	m.prevData = MutexWaitData{
		timestamp: time.Now(),
		total:     total,
	}

	ticker := time.NewTicker(m.config.MonitorInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			total, err := m.getMutexWaitTotal()
			if err != nil {
				log.Error("Error collecting mutex wait total", zap.Error(err))
				continue
			}

			currentData := MutexWaitData{
				timestamp: now,
				total:     total,
			}

			lockWaitRate := m.calculateDelta(m.prevData, currentData)
			log.Info("Lock wait rate status",
				zap.Float64("rate_seconds_per_second", lockWaitRate),
				zap.Int("current_gomaxprocs", m.currentProcs))

			// Update previous data
			m.prevData = currentData

			// Check cooldown
			if now.Sub(m.lastAdjustment) < m.config.AdjustmentCooldown {
				log.Info("Adjustment cooldown period. No changes made")
				continue
			}

			// Decide whether to adjust GOMAXPROCS
			if lockWaitRate > m.config.HighLockWaitRateThreshold && m.currentProcs > m.config.MinProcs {
				newProcs := m.currentProcs - 1
				runtime.GOMAXPROCS(newProcs)
				m.currentProcs = newProcs
				m.lastAdjustment = now
				log.Info("Decreased GOMAXPROCS due to high lock contention",
					zap.Int("new_procs", newProcs))
			} else if lockWaitRate < m.config.LowLockWaitRateThreshold && m.currentProcs +2  < m.config.MaxProcs {
				newProcs := m.currentProcs + 2
				runtime.GOMAXPROCS(newProcs)
				m.currentProcs = newProcs
				m.lastAdjustment = now
				log.Info("Increased GOMAXPROCS due to low lock contention",
					zap.Int("new_procs", newProcs))
			} else {
				log.Info("No adjustment needed, lock wait rate", zap.Float64("rate", lockWaitRate), zap.Int("current_gomaxprocs", m.currentProcs))
			}
		}
	}
}

// calculateDelta computes the rate of mutex wait time between two measurements.
func (m *Manager) calculateDelta(prev, current MutexWaitData) float64 {
	elapsed := current.timestamp.Sub(prev.timestamp).Seconds()
	if elapsed <= 0 {
		return 0
	}
	return (current.total - prev.total) / elapsed
}

// Descriptor for /sync/mutex/wait/total:seconds
var mutexWaitTotalDesc = "/sync/mutex/wait/total:seconds"

// getMutexWaitTotal retrieves the current value of /sync/mutex/wait/total:seconds
func (m *Manager) getMutexWaitTotal() (float64, error) {
	sample := make([]metrics.Sample, 1)
	sample[0].Name = mutexWaitTotalDesc
	metrics.Read(sample)

	switch v := sample[0].Value.Kind(); v {
	case metrics.KindFloat64:
		return sample[0].Value.Float64(), nil
	default:
		return 0, fmt.Errorf("unexpected metric value type %v for %s", v, mutexWaitTotalDesc)
	}
}
