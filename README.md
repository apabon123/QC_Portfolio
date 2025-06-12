# QC Portfolio - Local Development for QuantConnect Cloud

A hybrid development environment for building and deploying algorithmic trading strategies to QuantConnect Cloud while maintaining full local development capabilities.

## 🎯 Project Philosophy

This repository structure is intentionally designed as a **hybrid approach** that bridges local Python development with QuantConnect's cloud execution environment. While not following standard Python project conventions, this structure optimizes for:

- **Local Development**: Full IDE support, debugging, version control
- **Cloud Deployment**: Seamless pushing to QuantConnect Cloud via LEAN CLI
- **File Size Management**: Organized components to respect QC's per-file limits
- **Multi-Project Support**: Scalable structure for multiple trading strategies

## 📁 Repository Structure

```
QC_Portfolio/
├── README.md                    # This overview document
├── .gitignore                   # Excludes dev files, backtests, cache
│
├── CTA Replication/             # Individual Trading Project
│   ├── src/                     # ← SYNCS TO QUANTCONNECT
│   │   ├── main.py              # Algorithm entry point
│   │   ├── research.ipynb       # Jupyter research notebook
│   │   ├── components/          # Core system components
│   │   ├── strategies/          # Individual strategy implementations
│   │   ├── risk/               # Risk management modules
│   │   └── utils/              # Helper utilities
│   ├── README.md               # Project-specific documentation
│   ├── TECHNICAL.md            # Technical implementation details
│   ├── config.json             # Project configuration
│   ├── backtests/              # Local backtest results (not synced)
│   ├── data/                   # Local data files (not synced)
│   └── .gitignore              # Project-specific exclusions
│
└── [Future Projects]/          # Additional trading strategies
    ├── src/                    # Each project follows same pattern
    ├── README.md
    └── ...
```

## 🚀 Development Workflow

### 1. **Local Development**
- Develop strategies in your preferred IDE (VS Code, PyCharm, etc.)
- Full debugging and testing capabilities
- Organize code by function (strategies, components, risk, utils)
- Keep documentation and research files locally

### 2. **QuantConnect Sync**
- Only the `/src` folder and its contents sync to QuantConnect
- `.py` and `.ipynb` files are automatically uploaded
- Folder structure is preserved in QC Cloud
- File size limits automatically respected (64KB per file with Researcher plan)

### 3. **Deploy to Cloud**
```bash
# Push individual project to QuantConnect
lean cloud push --project "CTA Replication"

# Commit changes to GitHub
git add .
git commit -m "Update strategy implementations"
git push origin main
```

## 📊 Current Projects

### **CTA Replication**
A sophisticated three-layer CTA (Commodity Trading Advisor) portfolio system implementing multiple momentum strategies:

- **Kestner CTA Strategy**: Lars Kestner's momentum replication methodology
- **MTUM CTA Strategy**: Multi-timeframe momentum with regime detection  
- **HMM CTA Strategy**: Hidden Markov Model-based trend following
- **Risk Management**: Multi-layer portfolio and position-level risk controls
- **Dynamic Allocation**: Volatility-targeted position sizing and rebalancing

**Status**: ✅ Active Development | 🚀 Deployed to QC Cloud

## 🛠️ Technical Requirements

### **Local Development**
- Python 3.8+
- LEAN CLI installed and configured
- QuantConnect account (Researcher plan recommended for 64KB file limits)

### **File Organization Principles**
- **Modularity**: Each component in separate files under size limits
- **Functionality**: Organized by purpose (strategies, risk, components, utils)
- **Maintainability**: Clear separation between local dev files and cloud-sync files
- **Scalability**: Easy to add new projects following established patterns

## 🔄 Sync Behavior

### **What Syncs to QuantConnect:**
- All `.py` files in `/src` folders
- All `.ipynb` Jupyter notebooks  
- Folder structure within `/src`
- Project configuration files

### **What Stays Local:**
- Documentation files (README.md, TECHNICAL.md)
- Backtest results and logs
- Data files and research outputs
- IDE configuration (.vscode, .idea)
- Python cache and temporary files

## 🎯 Why This Structure?

### **Advantages of Hybrid Approach:**
1. **Best of Both Worlds**: Local development flexibility + cloud execution power
2. **File Size Compliance**: Automatic adherence to QuantConnect limits
3. **Clean Separation**: Development artifacts don't clutter cloud environment  
4. **Version Control**: Full Git history for code and documentation
5. **Multi-Project Support**: Easy to scale to additional trading strategies
6. **IDE Integration**: Full IntelliSense, debugging, and refactoring support

### **Trade-offs:**
- Non-standard Python project structure
- Requires understanding of sync behavior
- LEAN CLI dependency for deployment

## 📈 Getting Started

1. **Clone Repository**
   ```bash
   git clone [repository-url]
   cd QC_Portfolio
   ```

2. **Configure LEAN CLI**
   ```bash
   lean login
   lean cloud status
   ```

3. **Develop Locally**
   - Open project in your preferred IDE
   - Modify files in `[Project]/src/` folders
   - Test and debug locally

4. **Deploy to Cloud**
   ```bash
   lean cloud push --project "[Project Name]"
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   git push origin main
   ```

## 🤝 Contributing

This repository follows a project-per-folder approach. When adding new trading strategies:

1. Create new project folder following `CTA Replication/` pattern
2. Implement strategy in organized `/src` structure  
3. Include project-specific README and documentation
4. Ensure all files respect QuantConnect size limits
5. Test deployment with `lean cloud push`

## 📚 Documentation

- Each project includes detailed README with implementation specifics
- TECHNICAL.md files contain architecture and algorithm details
- Code comments focus on trading logic and mathematical implementations
- Git history provides development timeline and decision rationale

---

**Note**: This hybrid structure optimizes for algorithmic trading development where local flexibility and cloud execution capabilities are both essential. While unconventional for standard Python projects, it maximizes productivity for QuantConnect-based trading system development.
