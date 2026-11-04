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
        self.node.goto_target(self.node.x, self.node.y, 7.0, 1.0)

    def start_rtb(self, base_x=-10.0, base_y=7.0):
        self.tx = base_x
        self.ty = base_y
        self.phase = "RTB_ASCEND"
        self.log(f"🏠 RTB: Ascending to safe altitude (7.0m)...")
        self.node.goto_target(self.node.x, self.node.y, 7.0, 1.0)

    def cancel_mission(self):
        self.phase = "IDLE"

    def update(self, current_state_str):
        if self.phase == "IDLE":
            return

        navigating = getattr(self.node, 'is_navigating', False)
        
        if not navigating and current_state_str == "TARGET REACHED":
            
            # --- LAUNCH SEQUENCE ---
            if self.phase == "ASCEND":
                self.phase = "TRANSIT"
                self.log(f"🚀 MISSION: Transiting to X:{self.tx}, Y:{self.ty}...")
                self.node.goto_target(self.tx, self.ty, 7.0, 3.0)
                
            elif self.phase == "TRANSIT":
                self.phase = "DESCEND"
                self.log("🚀 MISSION: Target reached. Descending to 2.0m...")
                self.node.goto_target(self.tx, self.ty, 2.0, 1.0)
                
            elif self.phase == "DESCEND":
                self.phase = "IDLE"
                self.log("🚀 MISSION: Descent complete. Commencing Survey.")
                self.run_survey()
                
            # --- RTB SEQUENCE ---
            elif self.phase == "RTB_ASCEND":
                self.phase = "RTB_TRANSIT"
                self.log(f"🏠 RTB: Transiting to Base ({self.tx}, {self.ty})...")
                self.node.goto_target(self.tx, self.ty, 7.0, 3.0)
                
            elif self.phase == "RTB_TRANSIT":
                self.phase = "RTB_DESCEND"
                self.log("🏠 RTB: Base reached. Descending to 2.0m...")
                self.node.goto_target(self.tx, self.ty, 2.0, 1.0)
                
            elif self.phase == "RTB_DESCEND":
                self.phase = "IDLE"
                self.log("🏠 RTB: Drone has successfully returned to Base.")