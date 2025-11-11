from assertpy import assert_that

from colors_of_meaning.infrastructure.system.health_checker import SystemHealthChecker


class TestSystemHealthChecker:
    def test_should_initialize_with_empty_checks(self):
        health_checker = SystemHealthChecker()

        result = health_checker.check_liveness()

        assert_that(result.is_healthy).is_true()

    def test_should_report_healthy_when_no_readiness_checks(self):
        health_checker = SystemHealthChecker()

        result = health_checker.check_readiness()

        assert_that(result.is_healthy).is_true()

    def test_should_register_liveness_check(self):
        health_checker = SystemHealthChecker()

        def check() -> bool:
            return True

        health_checker.register_liveness_check(check)

        result = health_checker.check_liveness()

        assert_that(result.is_healthy).is_true()

    def test_should_report_unhealthy_when_liveness_check_fails(self):
        health_checker = SystemHealthChecker()

        def check() -> bool:
            return False

        health_checker.register_liveness_check(check)

        result = health_checker.check_liveness()

        assert_that(result.is_healthy).is_false()

    def test_should_report_healthy_when_all_liveness_checks_pass(self):
        health_checker = SystemHealthChecker()

        def check1() -> bool:
            return True

        def check2() -> bool:
            return True

        health_checker.register_liveness_check(check1)
        health_checker.register_liveness_check(check2)

        result = health_checker.check_liveness()

        assert_that(result.is_healthy).is_true()

    def test_should_report_unhealthy_when_any_liveness_check_fails(self):
        health_checker = SystemHealthChecker()

        def check1() -> bool:
            return True

        def check2() -> bool:
            return False

        health_checker.register_liveness_check(check1)
        health_checker.register_liveness_check(check2)

        result = health_checker.check_liveness()

        assert_that(result.is_healthy).is_false()

    def test_should_register_readiness_check(self):
        health_checker = SystemHealthChecker()

        def check():
            return {"test": {"status": True}}

        health_checker.register_readiness_check(check)

        result = health_checker.check_readiness()

        assert_that(result.is_healthy).is_true()

    def test_should_report_unhealthy_when_readiness_check_fails(self):
        health_checker = SystemHealthChecker()

        def check():
            return {"test": {"status": False}}

        health_checker.register_readiness_check(check)

        result = health_checker.check_readiness()

        assert_that(result.is_healthy).is_false()

    def test_should_report_healthy_when_all_readiness_checks_pass(self):
        health_checker = SystemHealthChecker()

        def check1():
            return {"test1": {"status": True}}

        def check2():
            return {"test2": {"status": True}}

        health_checker.register_readiness_check(check1)
        health_checker.register_readiness_check(check2)

        result = health_checker.check_readiness()

        assert_that(result.is_healthy).is_true()

    def test_should_report_unhealthy_when_any_readiness_check_fails(self):
        health_checker = SystemHealthChecker()

        def check1():
            return {"test1": {"status": True}}

        def check2():
            return {"test2": {"status": False}}

        health_checker.register_readiness_check(check1)
        health_checker.register_readiness_check(check2)

        result = health_checker.check_readiness()

        assert_that(result.is_healthy).is_false()

    def test_should_combine_details_from_all_readiness_checks(self):
        health_checker = SystemHealthChecker()
        expected_details = {
            "test1": {"status": True, "message": "Test 1 passed"},
            "test2": {"status": True, "message": "Test 2 passed"},
        }

        def check1():
            return {"test1": {"status": True, "message": "Test 1 passed"}}

        def check2():
            return {"test2": {"status": True, "message": "Test 2 passed"}}

        health_checker.register_readiness_check(check1)
        health_checker.register_readiness_check(check2)

        result = health_checker.check_readiness()

        assert_that(result.details).is_equal_to(expected_details)

    def test_should_be_true_when_all_checks_pass(self):
        health_checker = SystemHealthChecker()
        details = {"service1": {"status": True}, "service2": {"status": True}}

        result = health_checker._are_all_checks_healthy(details)

        assert_that(result).is_true()

    def test_should_be_false_when_any_checks_fails(self):
        health_checker = SystemHealthChecker()
        details = {"service1": {"status": True}, "service2": {"status": False}}

        result = health_checker._are_all_checks_healthy(details)

        assert_that(result).is_false()
