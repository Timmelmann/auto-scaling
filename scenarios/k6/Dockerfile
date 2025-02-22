FROM grafana/k6:latest

# Copy load testing script
COPY scenario_script.js /scripts/scenario_script.js

# Set working directory
WORKDIR /scripts

# Default command
ENTRYPOINT ["k6"]
CMD ["run", "scenario_script.js"]