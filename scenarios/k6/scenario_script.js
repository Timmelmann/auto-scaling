import http from "k6/http";
import { sleep, check } from "k6";
import { Rate, Trend } from "k6/metrics";

// Get API domain from environment variable
const API_DOMAIN = __ENV.API_DOMAIN || "http://localhost:64521";

// Custom metrics
const errorRate = new Rate("error_rate");
const successRate = new Rate("success_rate");
const requestDuration = new Trend("request_duration");

// Real product IDs from the catalog
const PRODUCT_IDS = [
  "OLJCESPC7Z", // Sunglasses
  "66VCHSJNUP", // Tank Top
  "1YMWWN1N4O", // Watch
  "L9ECAV7KIM", // Loafers
  "2ZYFJ3GM2N", // Hairdryer
  "0PUK6V6EV0", // Candle Holder
  "LS4PSXUNUM", // Salt & Pepper Shakers
  "9SIQT8TOJO", // Bamboo Glass Jar
  "6E92ZMYYFZ", // Mug
];

// Helper function to get random product ID
function getRandomProductId() {
  return PRODUCT_IDS[Math.floor(Math.random() * PRODUCT_IDS.length)];
}

// API calls definition with dynamic domain
const API_CALLS = {
  getProducts: () => {
    const response = http.get(`${API_DOMAIN}/`);
    checkResponse(response, "getProducts");
    sleep(Math.random() * 2 + 1);
  },
  getProduct: () => {
    const productId = getRandomProductId();
    const response = http.get(`${API_DOMAIN}/product/${productId}`);
    checkResponse(response, "getProduct");
    sleep(Math.random() * 1.5 + 0.5);
  },
  checkout: () => {
    const response = http.post(`${API_DOMAIN}/cart/checkout`, {
      city: "Mountain View",
      country: "United States",
      credit_card_cvv: "672",
      credit_card_expiration_month: "5",
      credit_card_expiration_year: "2027",
      credit_card_number: "4432801561520454",
      email: "someOne@example.de",
      state: "CA",
      street_address: "elmstreet",
      zip_code: "94043",
    });
    checkResponse(response, "checkout");
    sleep(Math.random() * 3 + 2);
  },
  addToCart: () => {
    const response = http.post(`${API_DOMAIN}/cart`, {
      product_id: getRandomProductId(),
      quantity: Math.floor(Math.random() * 3) + 1,
    });
    checkResponse(response, "addToCart");
    sleep(Math.random() * 1 + 1);
  },
  getCart: () => {
    const response = http.get(`${API_DOMAIN}/cart`);
    checkResponse(response, "getCart");
    sleep(Math.random() * 1 + 1);
  },
  emptyCart: () => {
    const response = http.post(`${API_DOMAIN}/cart/empty`, {});
    checkResponse(response, "emptyCart");
    sleep(Math.random() * 1 + 1);
  },
  logout: () => {
    const response = http.get(`${API_DOMAIN}/logout`);
    checkResponse(response, "logout");
    sleep(Math.random() * 1 + 1);
  },
};

// User behavior patterns
const userFlows = {
  browserFlow: () => {
    API_CALLS.getProducts();
      API_CALLS.getProduct();
    if (Math.random() < 0.8) {
      API_CALLS.addToCart();
    }
    if (Math.random() < 0.7) {
      API_CALLS.getProducts();
    }
    if (Math.random() < 0.7) {
      API_CALLS.getProduct();
    }
    if (Math.random() < 0.5) {
      API_CALLS.getCart();
    }
    if (Math.random() < 0.7) {
      API_CALLS.getProduct();
    }
    if (Math.random() < 0.7) {
      API_CALLS.addToCart();
    }
    if (Math.random() < 0.7) {
      API_CALLS.getCart();
    }
    if (Math.random() < 0.4) {
      API_CALLS..checkout();
    }
    API_CALLS.emptyCart();
    API_CALLS.logout();
  },

  buyerFlow: () => {
    API_CALLS.getProduct();
    API_CALLS.addToCart();
    API_CALLS.getCart();
    API_CALLS.checkout();
    API_CALLS.emptyCart();
    API_CALLS.logout();
  },

  apiConsumerFlow: () => {
    for (let i = 0; i < 3; i++) {
      API_CALLS.getProducts();
      API_CALLS.getProduct();
    }
  },
};

// Helper function for response checking
function checkResponse(response, apiName) {
  const success = check(response, {
    "status is 200": (r) => r.status === 200,
    "response time < 500ms": (r) => r.timings.duration < 500,
  });
  successRate.add(success, { api: apiName });
  errorRate.add(!success, { api: apiName });
  requestDuration.add(response.timings.duration, { api: apiName });
}

// Day/Night cycle configuration
const DAY_CYCLE = {
  PEAK_HOURS: {
    browserUsers: 250,
    buyers: 200,
    apiRate: 100
  },
  NORMAL_HOURS: {
    browserUsers: 150,
    buyers: 120,
    apiRate: 100,
  },
  OFF_HOURS: {
    browserUsers: 30,
    buyers: 20,
    apiRate: 30,
  },
};

// Function to generate stages for multiple days
function generateMultiDayStages(numDays, cycleConfig) {
  const stages = [];
  for (let day = 0; day < numDays; day++) {
    stages.push(
      { duration: "6h", target: cycleConfig.OFF_HOURS }, // Night
      { duration: "4h", target: cycleConfig.NORMAL_HOURS }, // Morning ramp-up
      { duration: "10h", target: cycleConfig.PEAK_HOURS }, // Peak hours
      { duration: "4h", target: cycleConfig.NORMAL_HOURS } // Evening wind-down
    );
  }
  return stages.flat();
}

// Main configuration
export const options = {
  scenarios: {
    browser_users: {
      executor: "ramping-vus",
      exec: "browserFlow",
      startVUs: DAY_CYCLE.OFF_HOURS.browserUsers,
      stages: generateMultiDayStages(7, {
        OFF_HOURS: DAY_CYCLE.OFF_HOURS.browserUsers,
        NORMAL_HOURS: DAY_CYCLE.NORMAL_HOURS.browserUsers,
        PEAK_HOURS: DAY_CYCLE.PEAK_HOURS.browserUsers,
      }),
    },
    buyers: {
      executor: "ramping-vus",
      exec: "buyerFlow",
      startVUs: DAY_CYCLE.OFF_HOURS.buyers,
      stages: generateMultiDayStages(7, {
        OFF_HOURS: DAY_CYCLE.OFF_HOURS.buyers,
        NORMAL_HOURS: DAY_CYCLE.NORMAL_HOURS.buyers,
        PEAK_HOURS: DAY_CYCLE.PEAK_HOURS.buyers,
      }),
    },
    api_consumers: {
      executor: "ramping-arrival-rate",
      exec: "apiConsumerFlow",
      startRate: DAY_CYCLE.OFF_HOURS.apiRate,
      timeUnit: "1m",
      preAllocatedVUs: 50,
      maxVUs: 200,
      stages: generateMultiDayStages(7, {
        OFF_HOURS: DAY_CYCLE.OFF_HOURS.apiRate,
        NORMAL_HOURS: DAY_CYCLE.NORMAL_HOURS.apiRate,
        PEAK_HOURS: DAY_CYCLE.PEAK_HOURS.apiRate,
      }),
    },
  },

  thresholds: {
    http_req_duration: ["p(95)<500"],
    error_rate: ["rate<0.1"],
    "http_req_duration{api:checkout}": ["p(95)<1000"],
    "error_rate{api:checkout}": ["rate<0.05"],
  },
};

// Export main functions
export function browserFlow() {
  userFlows.browserFlow();
}

export function buyerFlow() {
  userFlows.buyerFlow();
}

export function apiConsumerFlow() {
  userFlows.apiConsumerFlow();
}

// Summary report
export function handleSummary(data) {
  const date = new Date().toISOString().split("T")[0];
  return {
    [`summary-${date}.json`]: JSON.stringify(
      {
        metrics: {
          http_reqs: data.metrics.http_reqs,
          error_rate: data.metrics.error_rate,
          success_rate: data.metrics.success_rate,
          vus: data.metrics.vus,
          request_duration: data.metrics.request_duration,
        },
        scenarios: {
          browser: data.metrics.scenarios.browser_users,
          buyers: data.metrics.scenarios.buyers,
          api: data.metrics.scenarios.api_consumers,
        },
      },
      null,
      2
    ),
  };
}
