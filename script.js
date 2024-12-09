

document.getElementById("prosthesisForm").addEventListener("submit", function (event) {
    event.preventDefault(); 
    const side = document.getElementById("side").value;
    const typeOfAmputation = document.getElementById("amputation").value;
  
    const recommendation = recommendProsthesis(side, typeOfAmputation);
  
    document.getElementById("recommendation").textContent = recommendation;
  });
  
  function recommendProsthesis(side, typeOfAmputation) {
    const prosthesisOptions = {
      "Abajo del codo": "The UnLimbited Arm v2.1 - Alfie Edition",
      "Arriba del codo": "Kwawu Arm 2.0 - Advanced Prosthetic",
      Mano: "Kwawu Arm 2.0 - Thermoform Version",
      "Congénita (de nacimiento)": "Custom Pediatric Prosthetic",
    };
  
    const prosthesis = prosthesisOptions[typeOfAmputation] || "Modelo estándar de prótesis";
  
    return `Recomendación: ${prosthesis} para el brazo ${side}.`;
  }
  