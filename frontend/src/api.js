import axios from "axios";
const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE });
export const getForecast = (siteId="site-A", hours=24) =>
  api.get("/forecast", { params: { site_id: siteId, horizon_hours: hours }})
     .then(r => r.data);
export const postBid = (payload) =>
  api.post("/bid", payload).then(r => r.data);
