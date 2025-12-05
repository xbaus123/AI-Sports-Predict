document.addEventListener('DOMContentLoaded', async () => {
    
    let playerNames = []; 

    // --- 1. Load Player Data & Team Data ---
    try {
        const playerRes = await fetch('/api/players');
        if (playerRes.ok) {
            playerNames = await playerRes.json();
            const dataList = document.getElementById('player-list');
            const fragment = document.createDocumentFragment();
            
            playerNames.forEach(player => {
                const option = document.createElement('option');
                option.value = player;
                fragment.appendChild(option);
            });
            dataList.appendChild(fragment);
        }

        const teamRes = await fetch('/api/teams');
        if (teamRes.ok) {
            const teams = await teamRes.json();
            const teamSelect = document.getElementById('opponentSelect');
            const fragment = document.createDocumentFragment();

            teams.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t;
                opt.textContent = t;
                fragment.appendChild(opt);
            });
            teamSelect.appendChild(fragment);
        }
    } catch (error) {
        console.error('Error loading data:', error);
    }

    // --- NEW: Auto-detect Team on Input ---
    const playerNameInput = document.getElementById('playerName');
    const teamDisplay = document.getElementById('playerTeamDisplay');

    if (playerNameInput) {
        playerNameInput.addEventListener('input', async () => {
            const name = playerNameInput.value;
            if (!playerNames.includes(name)) {
                teamDisplay.textContent = "Select a player...";
                teamDisplay.style.color = "#555";
                teamDisplay.style.fontWeight = "normal";
                return;
            }

            teamDisplay.textContent = "Detecting team...";
            teamDisplay.style.color = "#555";
            
            try {
                const res = await fetch(`/api/player/${encodeURIComponent(name)}`);
                if (res.ok) {
                    const data = await res.json();
                    if (data.team) {
                        teamDisplay.textContent = data.team; 
                        teamDisplay.style.color = "#1e3c72";
                        teamDisplay.style.fontWeight = "bold";
                        teamDisplay.style.fontSize = "1.1em";
                    } else {
                        teamDisplay.textContent = "Team not available";
                    }
                } else {
                    teamDisplay.textContent = "Player not found";
                    teamDisplay.style.color = "#dc3545";
                }
            } catch (e) {
                teamDisplay.textContent = "Error fetching team";
            }
        });
    }

    // --- 2. Prediction Form Handler ---
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const playerName = document.getElementById('playerName').value;
            const spread = document.getElementById('spread').value;
            const opponent = document.getElementById('opponentSelect').value;
            const location = document.querySelector('input[name="location"]:checked').value;

            const predictionResult = document.getElementById('predictionResult');
            const predictionError = document.getElementById('predictionError');
            const predictionLoading = document.getElementById('predictionLoading');
            const factorsList = document.getElementById('factorsList');

            predictionResult.classList.remove('show', 'over', 'under');
            predictionError.classList.remove('show');
            predictionLoading.classList.add('show');
            factorsList.innerHTML = ''; // Clear previous factors

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        player: playerName,
                        opponent: opponent,
                        location: location,
                        spread: spread
                    })
                });

                const data = await response.json();
                predictionLoading.classList.remove('show');

                if (response.ok) {
                    const pick = data.pick.toUpperCase();
                    document.getElementById('pickText').textContent = `Recommendation: ${pick}`;
                    document.getElementById('projectedPoints').textContent = data.projected_points;
                    
                    // Display Confidence
                    const confVal = document.getElementById('confidenceValue');
                    confVal.textContent = data.confidence;
                    
                    document.getElementById('edgeValue').textContent = `${data.edge} pts`;
                    document.getElementById('confidenceNote').textContent = data.confidence_note;

                    // Populate Factors
                    if (data.top_factors && data.top_factors.length > 0) {
                        data.top_factors.forEach(factor => {
                            const li = document.createElement('li');
                            li.innerHTML = `• <strong>${factor.name}</strong> (${(factor.score * 100).toFixed(1)}% influence)`;
                            li.style.marginBottom = "4px";
                            factorsList.appendChild(li);
                        });
                    }

                    predictionResult.classList.add('show', pick.toLowerCase());
                } else {
                    predictionError.textContent = data.error || 'An error occurred';
                    predictionError.classList.add('show');
                }
            } catch (error) {
                predictionLoading.classList.remove('show');
                predictionError.textContent = 'Network error: ' + error.message;
                predictionError.classList.add('show');
            }
        });
    }

    // --- 3. Stats Form Handler ---
    const statsForm = document.getElementById('statsForm');
    if (statsForm) {
        statsForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const playerName = document.getElementById('statsPlayerName').value;
            const statsResult = document.getElementById('statsResult');
            const statsError = document.getElementById('statsError');
            const statsLoading = document.getElementById('statsLoading');

            statsResult.style.display = 'none';
            statsError.classList.remove('show');
            statsLoading.classList.add('show');

            try {
                const response = await fetch(`/api/player/${encodeURIComponent(playerName)}`);
                const data = await response.json();
                
                statsLoading.classList.remove('show');

                if (response.ok) {
                    document.getElementById('statPts').textContent = data.pts;
                    document.getElementById('statReb').textContent = data.reb;
                    document.getElementById('statAst').textContent = data.ast;
                    document.getElementById('statMin').textContent = data.min;
                    document.getElementById('statFgPct').textContent = (data.fg_pct * 100).toFixed(1) + '%';
                    document.getElementById('statOpp').textContent = data.opp;
                    statsResult.style.display = 'grid';
                } else {
                    statsError.textContent = data.error || 'Player not found';
                    statsError.classList.add('show');
                }
            } catch (error) {
                statsLoading.classList.remove('show');
                statsError.textContent = 'Network error: ' + error.message;
                statsError.classList.add('show');
            }
        });
    }
});