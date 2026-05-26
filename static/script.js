async function predict() {
    const ship_type = document.getElementById("ship_type").value;
    const fuel_type = document.getElementById("fuel_type").value;
    const distance = parseFloat(document.getElementById("distance").value);
    const actual_fuel_input = document.getElementById("actual_fuel").value;
    
    if (isNaN(distance)) {
        alert("Please enter a valid distance.");
        return;
    }
    
    const payload = {
        ship_type: ship_type,
        fuel_type: fuel_type,
        distance: distance
    };
    
    if (actual_fuel_input) {
        payload.actual_fuel = parseFloat(actual_fuel_input);
    }

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            let predicted = data.predicted_fuel;
            let html = `<strong>Predicted Fuel:</strong> ${predicted} L`;
            
            if (data.actual_fuel !== undefined) {
                let actual = data.actual_fuel;
                let diff = data.difference;
                let diffPercent = ((actual - predicted) / predicted) * 100;
                let absDiffPercent = Math.abs(diffPercent);
                
                html += `<br><strong>Actual Fuel:</strong> ${actual} L`;
                html += `<br><strong>Difference (L):</strong> ${diff} L`;
                html += `<br><strong>Difference (%):</strong> ${diffPercent.toFixed(2)}%`;
                
                if (absDiffPercent > 10) {
                    html += `<div class="anomaly">⚠️ Anomalous fuel consumption detected! Deviation exceeds 10%.</div>`;
                }
            }
            document.getElementById("result").innerHTML = html;
        } else {
            document.getElementById("result").innerHTML = `<span style="color:red">Error: ${data.detail}</span>`;
        }
    } catch (err) {
        document.getElementById("result").innerHTML = `<span style="color:red">Request failed.</span>`;
    }
}