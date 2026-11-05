class MissionController:
    def __init__(self, node, log_callback, survey_callback):
        self.node = node
        self.log = log_callback
        self.run_survey = survey_callback
        
        self.phase = "IDLE"
        self.tx = 0.0
        self.ty = 0.0

    def start_mission(self, tx, ty):
        self.tx = tx
        self.ty = ty
        self.phase = "ASCEND"
        self.log(f"🚀 MISSION: Ascending to safe altitude (7.0m)...")
        self.node.goto_target(self.node.x, self.node.y, 7.0, 2.0)

    def start_rtb(self, base_x=-10.0, base_y=7.0):
        self.tx = base_x
        self.ty = base_y
        self.phase = "RTB_ASCEND"
        self.log(f"🏠 RTB: Ascending to safe altitude (7.0m)...")
        self.node.goto_target(self.node.x, self.node.y, 7.0, 2.0)

    def cancel_mission(self):
        self.phase = "IDLE"

    def update(self, current_state_str):
        if self.phase == "IDLE":
            return

        # --- DYNAMIC HAZARD OVERRIDE ---
        # Continuously monitor for hazards during approach or descent
        if self.phase in ["TRANSIT", "DESCEND"]:
            if getattr(self.node, 'fire_thermal', False) or getattr(self.node, 'fire_rgb', False):
                self.log("⚠️ HAZARD DETECTED: Fire identified! Aborting approach.")
                self.phase = "HAZARD_STANDOFF"
                # Stop horizontal movement instantly and climb/hover at a safe 7.0m altitude
                self.node.goto_target(self.node.x, self.node.y, 7.0, 2.0)
                return
            elif getattr(self.node, 'flood_detected', False):
                self.log("⚠️ HAZARD DETECTED: Flood identified! Aborting approach.")
                self.phase = "HAZARD_STANDOFF"
                # Stop horizontal movement instantly and climb/hover at a safe 7.0m altitude
                self.node.goto_target(self.node.x, self.node.y, 7.0, 2.0)
                return

        navigating = getattr(self.node, 'is_navigating', False)
        
        if not navigating and current_state_str == "TARGET REACHED":
            
            # --- LAUNCH SEQUENCE ---
            if self.phase == "ASCEND":
                self.phase = "TRANSIT"
                self.log(f"🚀 MISSION: Transiting to X:{self.tx}, Y:{self.ty}...")
                self.node.goto_target(self.tx, self.ty, 7.0, 5.0)
                
            elif self.phase == "TRANSIT":
                self.phase = "DESCEND"
                self.log("🚀 MISSION: Target reached. Descending to 2.0m...")
                self.node.goto_target(self.tx, self.ty, 2.0, 2.0)
                
            elif self.phase == "DESCEND":
                self.phase = "IDLE"
                self.log("🚀 MISSION: Descent complete. Commencing Survey.")
                self.run_survey()

            # --- HAZARD SEQUENCE ---
            elif self.phase == "HAZARD_STANDOFF":
                self.phase = "IDLE"
                self.log("⚠️ SAFE ALTITUDE REACHED. Commencing aerial survey from safe distance.")
                self.run_survey()
                
            # --- RTB SEQUENCE ---
            elif self.phase == "RTB_ASCEND":
                self.phase = "RTB_TRANSIT"
                self.log(f"🏠 RTB: Transiting to Base ({self.tx}, {self.ty})...")
                self.node.goto_target(self.tx, self.ty, 7.0, 5.0)
                
            elif self.phase == "RTB_TRANSIT":
                self.phase = "RTB_DESCEND"
                self.log("🏠 RTB: Base reached. Descending to 2.0m...")
                self.node.goto_target(self.tx, self.ty, 2.0, 2.0)
                
            elif self.phase == "RTB_DESCEND":
                self.phase = "IDLE"
                self.log("🏠 RTB: Drone has successfully returned to Base.")